/* mimic_ctrl.c - make the DH116 robot hand mimic the operator's hand.
 *
 *   sudo ./mimic_ctrl <ifname> [options]     drive the hand over EtherCAT
 *        ./mimic_ctrl --dry-run [options]    the same without EtherCAT (to test the GUI)
 *
 * hand_cam sends hand landmarks (mimic_msg.h) to 127.0.0.1:<port>. hand_math.c turns them into a closure
 * 0..1 per joint, which becomes a target 0..10000 for the hand (dh116.c). A new target is only sent when
 * a joint moved more than --deadband. One 10 ms loop: read all waiting UDP packets, then one bus cycle.
 *
 * Hand out of view or GUI silent: the hand holds its pose. Quit (GUI Q/Esc, Ctrl+C, SIGTERM): the hand
 * opens, the joints are disabled, the bus goes to INIT. Hand not answering: bus to INIT, exit code 2.
 */
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "dh116.h"
#include "hand_math.h"
#include "mimic_msg.h"

static_assert(HM_JOINTS == DH116_JOINTS, "hand_math and dh116 joint counts differ");
static_assert(MSG_CAL_OPEN == HM_CAL_OPEN && MSG_CAL_FIST == HM_CAL_FIST && MSG_CAL_RESET == HM_CAL_RESET,
              "calibration kinds differ");

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int s)
{
    (void)s;
    g_stop = 1;
}

typedef struct
{
    int sock;
    hm_tracker tracker;
    int wanted[DH116_JOINTS];              /* target from the latest usable hand */
    int quit;                              /* the GUI asked to quit */
    unsigned packets, hands_seen, dropped; /* for the log line */
} ctrl;

static void send_status(const ctrl *c, const dh116 *h, const struct sockaddr_in *to)
{
    status_msg s;
    memset(&s, 0, sizeof(s));
    s.version = MSG_VERSION;
    s.dry_run = (uint8_t)h->dry;
    s.got_feedback = (uint8_t)h->got_feedback;
    for (int j = 0; j < DH116_JOINTS; ++j)
    {
        s.target[j] = (int16_t)h->target[j];
        s.measured[j] = (int16_t)h->measured[j];
    }
    snprintf(s.text, sizeof(s.text), "%s", c->tracker.msg);
    sendto(c->sock, &s, sizeof(s), 0, (const struct sockaddr *)to, sizeof(*to));
}

/* Read every waiting packet and answer each one. All of them go through the filter in order;
 * the last usable hand sets c->wanted. */
static void read_packets(ctrl *c, const dh116 *h)
{
    landmarks_msg m;
    struct sockaddr_in from;
    for (;;)
    {
        socklen_t from_len = sizeof(from);
        double closure[DH116_JOINTS];
        ssize_t n = recvfrom(c->sock, &m, sizeof(m), MSG_DONTWAIT | MSG_TRUNC, (struct sockaddr *)&from, &from_len);
        if (n < 0) return; /* nothing left */
        if (n != (ssize_t)sizeof(m) || m.version != MSG_VERSION || !isfinite(m.time_s) || m.calibrate > MSG_CAL_RESET)
        {
            c->dropped++;
            continue;
        }
        if (m.flags & MSG_QUIT)
        {
            c->quit = 1;
            continue;
        }
        c->packets++;
        if (hm_tracker_update(&c->tracker, (m.flags & MSG_HAND_SEEN) ? (const float (*)[3])m.world : NULL,
                              (m.flags & MSG_RIGHT_HAND) != 0, m.time_s, closure))
        {
            c->hands_seen++;
            for (int j = 0; j < DH116_JOINTS; ++j) c->wanted[j] = (int)lround(closure[j] * DH116_POS_MAX);
        }
        if (m.calibrate != MSG_CAL_NONE)
        {
            hm_tracker_calibrate(&c->tracker, m.calibrate, m.time_s);
            printf("%s\n", c->tracker.msg);
            fflush(stdout);
        }
        send_status(c, h, &from);
    }
}

static int usage(const char *p)
{
    printf("usage: sudo %s <ifname> | %s --dry-run\n"
           "  --port N           UDP port (default %d)\n"
           "  --current 50..1000 current limit, per mille (default: not sent, the hand uses its own)\n"
           "  --speed 100..20000 joint speed (default 20000)\n"
           "  --deadband 0..2000 raw change needed before a new target is sent (default 50)\n"
           "  --calibration FILE (default hand_calibration.json)\n"
           "  --min-cutoff HZ    One-Euro filter minimum cutoff (default 1.0)\n"
           "  --beta B           One-Euro filter speed coefficient (default 1.0)\n"
           "  --quiet            no log line every second\n",
           p, p, MSG_PORT);
    return 1;
}

int main(int argc, char **argv)
{
    static dh116 hand; /* static: the SOEM context is large */
    static ctrl c;
    const char *ifname = NULL, *cal_path = "hand_calibration.json";
    int dry = 0, verbose = 1, port = MSG_PORT, current = 0, speed = 20000, deadband = 50;
    double min_cutoff = 1.0, beta = 1.0;
    unsigned moves = 0;
    char err[160], current_text[16];
    struct sockaddr_in addr;
    struct timespec last_log, now;

    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        int has_value = i + 1 < argc;
        if (!strcmp(a, "--dry-run")) dry = 1;
        else if (!strcmp(a, "--quiet")) verbose = 0;
        else if (!strcmp(a, "--port") && has_value) port = atoi(argv[++i]);
        else if (!strcmp(a, "--current") && has_value) current = atoi(argv[++i]);
        else if (!strcmp(a, "--speed") && has_value) speed = atoi(argv[++i]);
        else if (!strcmp(a, "--deadband") && has_value) deadband = atoi(argv[++i]);
        else if (!strcmp(a, "--calibration") && has_value) cal_path = argv[++i];
        else if (!strcmp(a, "--min-cutoff") && has_value) min_cutoff = atof(argv[++i]);
        else if (!strcmp(a, "--beta") && has_value) beta = atof(argv[++i]);
        else if (a[0] != '-' && !ifname) ifname = a;
        else return usage(argv[0]);
    }
    if (dry == (ifname != NULL)) return usage(argv[0]); /* need exactly one of --dry-run or an interface */
    if (port < 1 || port > 65535 || (current && (current < 50 || current > 1000)) || speed < 100 ||
        speed > 20000 || deadband < 0 || deadband > 2000 || !(min_cutoff > 0) || !(beta >= 0))
        return usage(argv[0]);

    if (hm_tracker_init(&c.tracker, cal_path, min_cutoff, beta, err, sizeof(err)))
    {
        fprintf(stderr, "calibration: %s\n", err);
        return 1;
    }

    c.sock = socket(AF_INET, SOCK_DGRAM, 0);
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); /* local only */
    if (c.sock < 0 || bind(c.sock, (struct sockaddr *)&addr, sizeof(addr)))
    {
        fprintf(stderr, "cannot listen on 127.0.0.1:%d: %s\n", port, strerror(errno));
        return 1;
    }

    /* Ctrl+C / SIGTERM only end the loop, so the hand still gets opened and disabled below. */
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    if (dh116_start(&hand, ifname, speed, current)) return 2;
    memcpy(c.wanted, hand.target, sizeof(c.wanted)); /* hold the current pose until a hand is seen */

    if (current) snprintf(current_text, sizeof(current_text), "%d", current);
    else snprintf(current_text, sizeof(current_text), "not sent");
    printf("mimic_ctrl %s: listening on 127.0.0.1:%d, speed %d, current %s, deadband %d, calibration %s (%s)\n",
           dry ? "DRY RUN (no EtherCAT)" : ifname, port, speed, current_text, deadband, cal_path, c.tracker.msg);
    if (verbose) printf("joint order: thumb sweep, thumb bend, index, middle, ring, little\n");
    fflush(stdout);

    clock_gettime(CLOCK_MONOTONIC, &last_log);
    while (!g_stop && !c.quit)
    {
        int changed = 0;
        read_packets(&c, &hand);
        for (int j = 0; j < DH116_JOINTS; ++j)
            if (abs(c.wanted[j] - hand.target[j]) > deadband) changed = 1;
        if (changed)
        {
            dh116_set_target(&hand, c.wanted);
            moves++;
        }
        if (dh116_cycle(&hand)) return 2;

        clock_gettime(CLOCK_MONOTONIC, &now);
        double dt = (now.tv_sec - last_log.tv_sec) + (now.tv_nsec - last_log.tv_nsec) * 1e-9;
        if (verbose && dt >= 1.0)
        {
            printf("%5.1f pkt/s %5.1f hands/s %4.1f moves/s drop %u | target:", c.packets / dt, c.hands_seen / dt,
                   moves / dt, c.dropped);
            for (int j = 0; j < DH116_JOINTS; ++j) printf(" %5d", hand.target[j]);
            if (hand.got_feedback)
            {
                printf(" | measured:");
                for (int j = 0; j < DH116_JOINTS; ++j) printf(" %5d", hand.measured[j]);
            }
            printf("\n");
            fflush(stdout);
            c.packets = c.hands_seen = moves = 0;
            last_log = now;
        }
    }

    printf("stopping: %s\n", c.quit ? "GUI quit" : "signal");
    if (!dry) printf("opening the hand ...\n");
    if (dh116_open_fingers(&hand)) return 2;
    dh116_stop(&hand);
    close(c.sock);
    return 0;
}
