/* dh116.c - see dh116.h.
 *
 * Process data, from DH116_DH116P_Protocol.xlsx (sheet ECAT). 16-bit values are little-endian.
 * Byte 0 of every 64-byte RPDO frame is the frame type, byte 1 the number of axes (or entries) that follow.
 *   0x00  select feedback: the list of TPDO frame types the hand should send back
 *   0x01  command:         per axis 2 bytes = mode, command
 *   0x05  data:            per axis 4 bytes = position, speed / 10
 *   0x0B  data:            per axis 6 bytes = position, speed / 10, current (per mille)
 * TPDO (192 bytes) frame type 0x00: per axis 6 bytes = position, speed, current.
 */
#include "dh116.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Axis k of a frame is joint WIRE_JOINT[k]: the hand has the thumb bend before the thumb sweep.
 * Checked on our L1 hand; the joint table in the vendor library disagrees. */
static const int WIRE_JOINT[DH116_JOINTS] = {1, 0, 2, 3, 4, 5};

/* A command only takes effect when it differs from the previous one. */
enum { CMD_IDLE = 0, CMD_ENABLE = 1, CMD_DISABLE = 2, CMD_MOVE = 8 /* "cancel e-stop" in the sheet */ };
#define MODE_POSITION 0x20 /* as sent by the vendor library: position mode (bits 0-2 = 0), bit 5 unknown */

/* Frames sent once after every target change, one per cycle; after them the data frame repeats.
 * Change this line for the hardware tests:
 *   {SEND_MOVE}               MOVE, then the new target (default)
 *   {SEND_TARGET, SEND_MOVE}  new target, MOVE, new target (the older, verified order)
 *   {SEND_TARGET}             no MOVE at all */
enum { SEND_TARGET, SEND_MOVE };
static const int AFTER_CHANGE[] = {SEND_MOVE};
#define N_AFTER_CHANGE ((int)(sizeof(AFTER_CHANGE) / sizeof(AFTER_CHANGE[0])))

#define WKC_OK 3              /* working counter of a good cycle: the hand read its outputs and wrote its inputs */
#define FAULT_MS 100          /* the hand not answering for this long is a fault */
#define OPEN_TIMEOUT_MS 2000  /* dh116_open_fingers gives up after this */
#define OPEN_MIN_MS 100       /* ... and sends the open target at least this long */
#define OPEN_TOLERANCE 150    /* raw position that counts as open */

static void put16(uint8_t *p, int v)
{
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
}

static int16_t get16(const uint8_t *p) { return (int16_t)(p[0] | (p[1] << 8)); }

/* ---------------------------------------------------------------- frames */

static void frame_select_feedback(uint8_t f[DH116_RPDO_BYTES])
{
    memset(f, 0, DH116_RPDO_BYTES);
    f[0] = 0x00;
    f[1] = 2;    /* two TPDO frame types: */
    f[2] = 0x00; /* joint feedback */
    f[3] = 0x40; /* not decoded yet */
}

/* The same command for every joint. */
static void frame_command(uint8_t f[DH116_RPDO_BYTES], int cmd)
{
    memset(f, 0, DH116_RPDO_BYTES);
    f[0] = 0x01;
    f[1] = DH116_JOINTS;
    for (int k = 0; k < DH116_JOINTS; ++k)
    {
        f[2 + 2 * k] = MODE_POSITION;
        f[3 + 2 * k] = (uint8_t)cmd;
    }
}

/* Rebuild h->data from h->target.
 * With a current limit: type 0x0B (position, speed, current), the frame the vendor library sends.
 * Without: type 0x05 (position, speed), so the hand uses its own current limit (untested on hardware). */
static void frame_data(dh116 *h)
{
    int with_current = h->current > 0, axis_bytes = with_current ? 6 : 4;
    memset(h->data, 0, sizeof(h->data));
    h->data[0] = with_current ? 0x0B : 0x05;
    h->data[1] = DH116_JOINTS;
    for (int k = 0; k < DH116_JOINTS; ++k)
    {
        uint8_t *axis = h->data + 2 + axis_bytes * k;
        put16(axis, h->target[WIRE_JOINT[k]]);
        put16(axis + 2, h->speed / 10);
        if (with_current) put16(axis + 4, h->current);
    }
}

/* Joint feedback (TPDO frame type 0x00); only the position is used. */
static void read_feedback(dh116 *h, const uint8_t *in)
{
    int n = in[1] < DH116_JOINTS ? in[1] : DH116_JOINTS;
    if (in[0] != 0x00) return;
    for (int k = 0; k < n; ++k) h->measured[WIRE_JOINT[k]] = get16(in + 2 + 6 * k);
    if (n > 0) h->got_feedback = 1;
}

/* ---------------------------------------------------------------- bus cycle */

/* Sleep until the next tick. More than a cycle late: restart the schedule instead of catching up. */
static void wait_next_cycle(struct timespec *next)
{
    struct timespec now;
    next->tv_nsec += DH116_CYCLE_MS * 1000000L;
    if (next->tv_nsec >= 1000000000L)
    {
        next->tv_nsec -= 1000000000L;
        next->tv_sec++;
    }
    clock_gettime(CLOCK_MONOTONIC, &now);
    if ((now.tv_sec - next->tv_sec) * 1000000000LL + (now.tv_nsec - next->tv_nsec) > DH116_CYCLE_MS * 1000000LL)
        *next = now;
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, next, NULL);
}

static void go_init_and_close(dh116 *h)
{
    h->ctx.slavelist[0].state = EC_STATE_INIT;
    ecx_writestate(&h->ctx, 0);
    ecx_close(&h->ctx);
}

/* Send frame f, read the hand's answer, wait for the next tick. -1 = hand lost (bus closed). */
static int exchange(dh116 *h, const uint8_t f[DH116_RPDO_BYTES])
{
    if (!h->dry)
    {
        ec_slavet *hand = &h->ctx.slavelist[1];
        memcpy(hand->outputs, f, DH116_RPDO_BYTES);
        ecx_send_processdata(&h->ctx);
        if (ecx_receive_processdata(&h->ctx, EC_TIMEOUTRET) >= WKC_OK)
        {
            h->missed_cycles = 0;
            read_feedback(h, hand->inputs);
        }
        else if (++h->missed_cycles >= FAULT_MS / DH116_CYCLE_MS)
        {
            fprintf(stderr, "FAULT: hand not answering for %d ms (working counter below %d). Bus -> INIT.\n",
                    FAULT_MS, WKC_OK);
            go_init_and_close(h);
            return -1;
        }
    }
    wait_next_cycle(&h->next_cycle);
    return 0;
}

/* Send the same frame for ms milliseconds. */
static int hold(dh116 *h, const uint8_t f[DH116_RPDO_BYTES], int ms)
{
    for (int t = 0; t < ms; t += DH116_CYCLE_MS)
        if (exchange(h, f)) return -1;
    return 0;
}

/* ---------------------------------------------------------------- public */

int dh116_start(dh116 *h, const char *ifname, int speed, int current)
{
    uint8_t f[DH116_RPDO_BYTES] = {0};
    ecx_contextt *ctx = &h->ctx;
    ec_slavet *hand;

    memset(h, 0, sizeof(*h));
    hand = &ctx->slavelist[1];
    h->speed = speed;
    h->current = current;
    h->step = N_AFTER_CHANGE; /* nothing to send until the first target change */
    clock_gettime(CLOCK_MONOTONIC, &h->next_cycle);
    if (!ifname)
    {
        h->dry = 1;
        frame_data(h);
        return 0;
    }

    printf("bus: init on '%s' ... ", ifname);
    if (!ecx_init(ctx, ifname))
    {
        printf("FAILED (need sudo? wrong interface?)\n");
        return -1;
    }
    if (ecx_config_init(ctx) <= 0 || hand->eep_man != 0x00017185 || hand->eep_id != 0x00660000)
    {
        printf("no DH116EC found as the first slave\n");
        ecx_close(ctx);
        return -1;
    }
    ecx_config_map_group(ctx, h->iomap, 0);
    ecx_configdc(ctx);
    if (hand->Obytes != DH116_RPDO_BYTES || hand->Ibytes != DH116_TPDO_BYTES)
    {
        printf("unexpected PDO sizes %u/%u\n", hand->Obytes, hand->Ibytes);
        ecx_close(ctx);
        return -1;
    }
    ecx_statecheck(ctx, 0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE * 4);

    /* Ask for OP while exchanging process data (a slave wants valid outputs before it enters OP). */
    if (exchange(h, f)) return -1;
    ctx->slavelist[0].state = EC_STATE_OPERATIONAL;
    ecx_writestate(ctx, 0);
    for (int i = 0; i < 20; ++i)
    {
        if (exchange(h, f)) return -1;
        ecx_statecheck(ctx, 0, EC_STATE_OPERATIONAL, EC_TIMEOUTSTATE / 10);
        if (ctx->slavelist[0].state == EC_STATE_OPERATIONAL) break;
    }
    ecx_readstate(ctx);
    if (hand->state != EC_STATE_OPERATIONAL)
    {
        printf("did not reach OP (AL 0x%04X)\n", hand->ALstatuscode);
        go_init_and_close(h);
        return -1;
    }

    /* Empty frames, select the feedback, then enable all joints via idle (commands act only on a change). */
    if (hold(h, f, 250)) return -1;
    frame_select_feedback(f);
    if (hold(h, f, 500)) return -1;
    frame_command(f, CMD_IDLE);
    if (hold(h, f, 100)) return -1;
    frame_command(f, CMD_ENABLE);
    if (hold(h, f, 200)) return -1;

    if (!h->got_feedback)
    {
        printf("OP, joints enabled, but no position feedback: refusing to move\n");
        dh116_stop(h);
        return -1;
    }
    printf("OP, joints enabled, feedback ok\n");

    /* Hold the pose the hand is in now. That is not a change, so no MOVE. */
    dh116_set_target(h, h->measured);
    h->step = N_AFTER_CHANGE;
    return 0;
}

void dh116_set_target(dh116 *h, const int pos[DH116_JOINTS])
{
    for (int j = 0; j < DH116_JOINTS; ++j)
        h->target[j] = pos[j] < 0 ? 0 : pos[j] > DH116_POS_MAX ? DH116_POS_MAX : pos[j];
    frame_data(h);
    h->step = 0; /* send the AFTER_CHANGE frames first */
}

int dh116_cycle(dh116 *h)
{
    uint8_t move[DH116_RPDO_BYTES];
    int send = h->step < N_AFTER_CHANGE ? AFTER_CHANGE[h->step++] : SEND_TARGET;
    if (send == SEND_MOVE)
    {
        frame_command(move, CMD_MOVE);
        return exchange(h, move);
    }
    return exchange(h, h->data);
}

int dh116_open_fingers(dh116 *h)
{
    static const int open[DH116_JOINTS] = {0};
    if (h->dry) return 0;
    dh116_set_target(h, open);
    for (int ms = 0; ms < OPEN_TIMEOUT_MS; ms += DH116_CYCLE_MS)
    {
        int is_open = ms >= OPEN_MIN_MS;
        if (dh116_cycle(h)) return -1;
        for (int j = 0; j < DH116_JOINTS; ++j)
            if (abs(h->measured[j]) > OPEN_TOLERANCE) is_open = 0;
        if (is_open) return 0;
    }
    printf("fingers not open after %d ms, disabling anyway\n", OPEN_TIMEOUT_MS);
    return 0;
}

void dh116_stop(dh116 *h)
{
    uint8_t f[DH116_RPDO_BYTES];
    if (h->dry) return;
    frame_command(f, CMD_DISABLE);
    if (hold(h, f, 100)) return; /* the hand stopped answering; the bus is already closed */
    go_init_and_close(h);
    printf("bus: joints disabled, INIT, closed\n");
}
