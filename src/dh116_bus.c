/* dh116_bus.c - see dh116_bus.h. */
#include "dh116_bus.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ts_add_us(struct timespec *t, long us)
{
   t->tv_nsec += us * 1000L;
   while (t->tv_nsec >= 1000000000L) { t->tv_nsec -= 1000000000L; t->tv_sec++; }
}

static void shutdown_init(dh116_bus *b)
{
   b->ctx.slavelist[0].state = EC_STATE_INIT;
   ecx_writestate(&b->ctx, 0);
   ecx_close(&b->ctx);
}

int dh116_bus_cycle(dh116_bus *b, const uint8_t rpdo[DH116_RPDO_BYTES])
{
   ec_slavet *s = b->ctx.slavelist + 1;
   struct timespec now;
   memcpy(s->outputs, rpdo, DH116_RPDO_BYTES);
   ecx_send_processdata(&b->ctx);
   b->wkc = ecx_receive_processdata(&b->ctx, EC_TIMEOUTRET);
   /* wkc 3 is normal for one hand; 20 bad cycles (100 ms) in a row = hand not answering */
   if (b->wkc < 3)
   {
      if (++b->bad_wkc >= 20)
      {
         fprintf(stderr, "FAULT: working counter below 3 for 20 cycles (hand not answering). Bus -> INIT.\n");
         shutdown_init(b);
         return -1;
      }
   }
   else
   {
      b->bad_wkc = 0;
      memcpy(b->tpdo, s->inputs, DH116_TPDO_BYTES);
      if (dh116_decode_joints(b->tpdo, &b->fb) > 0) b->have_fb = 1;
   }
   /* fixed period: sleep to an absolute deadline; if we fell behind, restart the schedule */
   ts_add_us(&b->next, DH116_CYCLE_US);
   clock_gettime(CLOCK_MONOTONIC, &now);
   if ((now.tv_sec - b->next.tv_sec) * 1000000000LL + (now.tv_nsec - b->next.tv_nsec) > DH116_CYCLE_US * 1000LL)
      b->next = now;
   clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &b->next, NULL);
   return 0;
}

int dh116_bus_hold(dh116_bus *b, const uint8_t rpdo[DH116_RPDO_BYTES], int n)
{
   while (n-- > 0)
      if (dh116_bus_cycle(b, rpdo)) return -1;
   return 0;
}

static void ctrl_all(uint8_t out[DH116_RPDO_BYTES], uint16_t word)
{
   uint16_t c[7];
   int j;
   for (j = 0; j < 7; ++j) c[j] = word;
   dh116_frame_ctrl(out, c);
}

int dh116_bus_open(dh116_bus *b, const char *ifname)
{
   ecx_contextt *ctx = &b->ctx;
   uint8_t f[DH116_RPDO_BYTES];
   int i;
   memset(b, 0, sizeof(*b));
   clock_gettime(CLOCK_MONOTONIC, &b->next);

   printf("bus: init on '%s' ... ", ifname);
   if (!ecx_init(ctx, ifname)) { printf("FAILED (need sudo? wrong interface?)\n"); return -1; }
   if (ecx_config_init(ctx) <= 0) { printf("NO SLAVES FOUND\n"); ecx_close(ctx); return -1; }
   if (ctx->slavelist[1].eep_man != 0x00017185 || ctx->slavelist[1].eep_id != 0x00660000)
   { printf("slave 1 is not a DH116EC\n"); ecx_close(ctx); return -1; }
   ecx_config_map_group(ctx, b->map, 0);
   ecx_configdc(ctx);
   if (ctx->slavelist[1].Obytes != DH116_RPDO_BYTES || ctx->slavelist[1].Ibytes != DH116_TPDO_BYTES)
   { printf("unexpected PDO sizes %u/%u\n", ctx->slavelist[1].Obytes, ctx->slavelist[1].Ibytes); ecx_close(ctx); return -1; }
   ecx_statecheck(ctx, 0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE * 4);

   memset(f, 0, sizeof(f));
   if (dh116_bus_cycle(b, f)) return -1;
   ctx->slavelist[0].state = EC_STATE_OPERATIONAL;
   ecx_writestate(ctx, 0);
   for (i = 0; i < 20; ++i)
   {
      if (dh116_bus_cycle(b, f)) return -1;
      ecx_statecheck(ctx, 0, EC_STATE_OPERATIONAL, EC_TIMEOUTSTATE / 10);
      if (ctx->slavelist[0].state == EC_STATE_OPERATIONAL) break;
   }
   ecx_readstate(ctx);
   if (ctx->slavelist[1].state != EC_STATE_OPERATIONAL)
   { printf("did not reach OP (AL 0x%04X)\n", ctx->slavelist[1].ALstatuscode); shutdown_init(b); return -1; }
   if (dh116_bus_hold(b, f, 50)) return -1;

   dh116_frame_init(f);
   if (dh116_bus_hold(b, f, 100)) return -1;
   ctrl_all(f, DH116_CTRL_ENABLE_PREP);
   if (dh116_bus_hold(b, f, 20)) return -1;
   ctrl_all(f, DH116_CTRL_ENABLED);
   if (dh116_bus_hold(b, f, 40)) return -1;
   printf("OP, joints enabled, feedback %s\n", b->have_fb ? "ok" : "MISSING");
   return 0;
}

int dh116_bus_move_wait(dh116_bus *b, const int pos[7], int vel, int cur, int max_cycles)
{
   uint8_t f[DH116_RPDO_BYTES];
   int v[7], c[7], j, i;
   uint16_t mv[7];
   for (j = 0; j < 7; ++j) { v[j] = vel; c[j] = cur; mv[j] = DH116_CTRL_MOVE; }
   dh116_frame_data(f, pos, v, c);
   if (dh116_bus_hold(b, f, 2)) return -1;
   dh116_frame_ctrl(f, mv);
   if (dh116_bus_cycle(b, f)) return -1;
   dh116_frame_data(f, pos, v, c);
   for (i = 0; i < max_cycles; ++i)
   {
      int done = b->have_fb;
      if (dh116_bus_cycle(b, f)) return -1;
      for (j = 1; j <= 6; ++j)
         if (abs(b->fb.pos[j] - pos[j]) > 150) done = 0;
      if (done && i > 20) break;
   }
   return 0;
}

void dh116_bus_close(dh116_bus *b)
{
   uint8_t f[DH116_RPDO_BYTES];
   ctrl_all(f, DH116_CTRL_DISABLED);
   if (dh116_bus_hold(b, f, 20)) return;
   shutdown_init(b);
   printf("bus: joints disabled, INIT, closed\n");
}
