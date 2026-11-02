/* dh116.h - drive one Leadshine DH116EC hand over EtherCAT (SOEM), or pretend to (dry run).
 *
 * Joint order everywhere: thumb sweep, thumb bend, index, middle, ring, little.
 * Joint positions are raw 0 (open) .. 10000 (closed).
 *
 * Usage: dh116_start, then call dh116_cycle every loop iteration (it sleeps until the next
 * 10 ms tick) followed by dh116_check_status, dh116_set_target whenever the wanted pose changes,
 * and finally dh116_open_fingers + dh116_stop (only dh116_stop after a joint fault).
 */
#ifndef DH116_H
#define DH116_H
#include <stdint.h>
#include <time.h>

#include "soem/soem.h"

#define DH116_JOINTS 6
#define DH116_CYCLE_MS 10
#define DH116_POS_MAX 10000
#define DH116_RPDO_BYTES 64  /* master -> hand */
#define DH116_TPDO_BYTES 192 /* hand -> master */

/* Status byte of a joint: bits 0-4 are flags, bits 5-7 the state (protocol sheet, ECAT, AXISX status). */
#define DH116_ST_ENABLED 0x01
#define DH116_ST_TORQUE 0x08 /* torque reached: the joint is at its current limit */
#define DH116_ST_HOMED 0x10  /* homing completed */
#define DH116_ST_STATE(s) ((s) >> 5)
enum { DH116_STOPPED, DH116_RUNNING, DH116_ALARM, DH116_LIMIT_POS, DH116_LIMIT_NEG, DH116_LIMIT_BOTH, DH116_ESTOP, DH116_HOMING };

typedef struct
{
    int dry;             /* 1 = no EtherCAT, dh116_cycle only keeps the 10 ms pace */
    ecx_contextt ctx;
    uint8 iomap[4096];   /* SOEM process image; this hand uses 64 + 192 bytes of it */

    int target[DH116_JOINTS];        /* last target given to dh116_set_target */
    int speed, current;              /* current <= 0: no current limit is sent */
    uint8_t data[DH116_RPDO_BYTES];  /* data frame for target, sent every cycle, rebuilt when the target changes */

    int measured[DH116_JOINTS];      /* position feedback from the hand */
    int measured_current[DH116_JOINTS]; /* current feedback, per mille of the rated current */
    uint8_t status[DH116_JOINTS];    /* status byte from the hand, see DH116_ST_* */
    uint8_t problem[DH116_JOINTS];   /* last problem printed by dh116_check_status, 0 = none */
    uint8_t settle[DH116_JOINTS];    /* cycles a changed warning has lasted, see dh116_check_status */
    int got_feedback;                /* 1 once any position feedback has arrived */
    int missed_cycles;               /* consecutive cycles the hand did not answer */
    struct timespec next_cycle;
} dh116;

/* Bring the hand up: INIT -> OP, select feedback, enable all joints, check their status, hold the current pose.
 * ifname NULL = dry run. Returns 0 when ready; on failure the bus is already closed. */
int dh116_start(dh116 *h, const char *ifname, int speed, int current);

/* New target pose; values are clamped to 0..DH116_POS_MAX. */
void dh116_set_target(dh116 *h, const int pos[DH116_JOINTS]);

/* One 10 ms cycle: send a frame, read feedback, wait for the next tick.
 * Returns -1 if the hand stopped answering (the bus is then already closed). */
int dh116_cycle(dh116 *h);

/* Print each joint whose problem changed. Fatal ones (alarm stop, emergency stop, not enabled) in red at once,
 * warnings (travel limit, homing, torque reached) in yellow once they have lasted 100 ms; then a line when it clears.
 * Returns -1 if a joint has a fatal problem: then stop without moving the hand. */
int dh116_check_status(dh116 *h);

/* Move all fingers to open and wait (up to 2 s) until they get there. -1 = bus fault. */
int dh116_open_fingers(dh116 *h);

/* Disable all joints, bus to INIT, close. */
void dh116_stop(dh116 *h);

#endif
