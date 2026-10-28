/* dh116.h - drive one Leadshine DH116EC hand over EtherCAT (SOEM), or pretend to (dry run).
 *
 * Joint order everywhere: thumb sweep, thumb bend, index, middle, ring, little.
 * Joint positions are raw 0 (open) .. 10000 (closed).
 *
 * Usage: dh116_start, then call dh116_cycle every loop iteration (it sleeps until the next
 * 10 ms tick), dh116_set_target whenever the wanted pose changes, and finally
 * dh116_open_fingers + dh116_stop.
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

typedef struct
{
    int dry;             /* 1 = no EtherCAT, dh116_cycle only keeps the 10 ms pace */
    ecx_contextt ctx;
    uint8 iomap[4096];   /* SOEM process image; this hand uses 64 + 192 bytes of it */

    int target[DH116_JOINTS];        /* last target given to dh116_set_target */
    int speed, current;              /* current <= 0: no current limit is sent */
    uint8_t data[DH116_RPDO_BYTES];  /* data frame for target, rebuilt only when the target changes */
    int step;                        /* progress through the frames sent after a target change */

    int measured[DH116_JOINTS];      /* position feedback from the hand */
    int got_feedback;                /* 1 once any position feedback has arrived */
    int missed_cycles;               /* consecutive cycles the hand did not answer */
    struct timespec next_cycle;
} dh116;

/* Bring the hand up: INIT -> OP, select feedback, enable all joints, hold the current pose.
 * ifname NULL = dry run. Returns 0 when ready; on failure the bus is already closed. */
int dh116_start(dh116 *h, const char *ifname, int speed, int current);

/* New target pose; values are clamped to 0..DH116_POS_MAX. */
void dh116_set_target(dh116 *h, const int pos[DH116_JOINTS]);

/* One 10 ms cycle: send a frame, read feedback, wait for the next tick.
 * Returns -1 if the hand stopped answering (the bus is then already closed). */
int dh116_cycle(dh116 *h);

/* Move all fingers to open and wait (up to 2 s) until they get there. -1 = bus fault. */
int dh116_open_fingers(dh116 *h);

/* Disable all joints, bus to INIT, close. */
void dh116_stop(dh116 *h);

#endif
