/* dh116_bus.h - EtherCAT bring-up / cycle / shutdown for one DH116EC, shared by
 * dh116_teleop and the hardware test.
 */
#ifndef DH116_BUS_H
#define DH116_BUS_H
#include <stdint.h>
#include <time.h>

#include "dh116_pdo.h"
#include "soem/soem.h"

#define DH116_CYCLE_US 5000

typedef struct
{
   ecx_contextt ctx;
   uint8 map[4096];
   dh116_feedback fb;
   int have_fb, wkc, bad_wkc;
   uint8_t tpdo[DH116_TPDO_BYTES];
   struct timespec next; /* absolute deadline of the next cycle */
} dh116_bus;

/* init -> OP -> init frame -> enable all joints. 0 = ready, otherwise the bus is already closed. */
int dh116_bus_open(dh116_bus *b, const char *ifname);
/* one process-data cycle at a fixed 5 ms period. 0 = ok, -1 = hand stopped answering (bus closed). */
int dh116_bus_cycle(dh116_bus *b, const uint8_t rpdo[DH116_RPDO_BYTES]);
/* repeat a frame n times; -1 on fault */
int dh116_bus_hold(dh116_bus *b, const uint8_t rpdo[DH116_RPDO_BYTES], int n);
/* move every joint to pos[1..6] (data, one MOVE, data) and wait up to max_cycles for it to arrive */
int dh116_bus_move_wait(dh116_bus *b, const int pos[7], int vel, int cur, int max_cycles);
/* disable all joints, INIT, close */
void dh116_bus_close(dh116_bus *b);

#endif
