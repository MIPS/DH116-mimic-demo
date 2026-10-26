/*
 * retarget_test.c - HARDWARE test: can the hand follow a stream of MOVE commands smoothly?
 * This is the assumption dh116_teleop rests on; run it before the first teleop session.
 *
 *   sudo ./build/retarget_test <ifname> [--joint 6] [--amp 6000] [--hz 0.5] [--rate 30]
 *                              [--pre 1] [--vel 20000] [--cur 1000] [--secs 6] [--csv /tmp/retarget.csv]
 *
 * Opens the hand, then streams a smooth 0..amp..0 cosine to ONE joint, updating the target --rate
 * times per second exactly like dh116_teleop (data frame x --pre, one MOVE frame, data frames).
 * Logs target and measured position every 5 ms to the CSV and prints the lag that best aligns
 * them and the remaining error. Then opens the hand, disables, INIT.
 *
 * Good: lag well under 100 ms, RMS error a few % of amp, and a smooth plot of the CSV.
 * Bad : the joint stalls/jerks on each new MOVE (staircase in the CSV) or does not move at all
 *       -> try --pre 20 (a longer data-frame preamble), then a lower --rate.
 */
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dh116_bus.h"

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int s) { (void)s; g_stop = 1; }

int main(int argc, char **argv)
{
   static dh116_bus bus;
   static int tgt_log[20000], pos_log[20000];
   const char *ifname = argc > 1 ? argv[1] : NULL, *csv = "/tmp/retarget.csv";
   int joint = 6, amp = 6000, rate = 30, pre = 1, vel = 20000, cur = 1000, i, j, n = 0, best = 0;
   double hz = 0.5, secs = 6, best_rms = 1e18;
   int zero[7] = {0}, sent[7] = {0}, vv[7], cc[7], phase = 0, pre_left = 0, every;
   uint16_t mv[7];
   uint8_t f[DH116_RPDO_BYTES];
   FILE *out;

   for (i = 2; i + 1 < argc; i += 2)
   {
      if (!strcmp(argv[i], "--joint")) joint = atoi(argv[i + 1]);
      else if (!strcmp(argv[i], "--amp")) amp = atoi(argv[i + 1]);
      else if (!strcmp(argv[i], "--hz")) hz = atof(argv[i + 1]);
      else if (!strcmp(argv[i], "--rate")) rate = atoi(argv[i + 1]);
      else if (!strcmp(argv[i], "--pre")) pre = atoi(argv[i + 1]);
      else if (!strcmp(argv[i], "--vel")) vel = atoi(argv[i + 1]);
      else if (!strcmp(argv[i], "--cur")) cur = atoi(argv[i + 1]);
      else if (!strcmp(argv[i], "--secs")) secs = atof(argv[i + 1]);
      else if (!strcmp(argv[i], "--csv")) csv = argv[i + 1];
      else ifname = NULL;
   }
   if (!ifname || ifname[0] == '-' || (argc - 2) % 2 || joint < 1 || joint > 6 || amp < 100 || amp > 10000 ||
       !(hz > 0 && hz <= 2) || rate < 1 || rate > 100 || pre < 1 || pre > 20 || vel < 100 || vel > 20000 ||
       cur < 50 || cur > 1000 || !(secs >= 1 && secs <= 60))
   {
      printf("usage: sudo %s <ifname> [--joint 1..6] [--amp 100..10000] [--hz 0..2] [--rate 1..100]\n"
             "       [--pre 1..20] [--vel 100..20000] [--cur 50..1000] [--secs 1..60] [--csv FILE]\n", argv[0]);
      return 1;
   }
   signal(SIGINT, on_signal);
   for (j = 0; j < 7; ++j) { vv[j] = vel; cc[j] = cur; mv[j] = DH116_CTRL_MOVE; }
   every = 200 / rate; /* cycles between target updates */
   if (every < pre + 1) every = pre + 1;

   if (dh116_bus_open(&bus, ifname)) return 2;
   printf("opening the hand ...\n");
   if (dh116_bus_move_wait(&bus, zero, vel, cur, 400)) return 2;

   printf("streaming joint %d: 0..%d at %.2f Hz, new target every %d ms, pre %d, vel %d, cur %d, %.0f s\n",
          joint, amp, hz, every * 5, pre, vel, cur, secs);
   for (i = 0; i < (int)(secs * 200) && !g_stop && n < 20000; ++i)
   {
      if (phase == 0 && i % every == 0)
      {
         sent[joint] = (int)lround(amp * 0.5 * (1.0 - cos(2.0 * M_PI * hz * i * 0.005)));
         phase = 1;
         pre_left = pre;
      }
      if (phase == 2) { dh116_frame_ctrl(f, mv); phase = 0; }
      else
      {
         dh116_frame_data(f, sent, vv, cc);
         if (phase == 1 && --pre_left == 0) phase = 2;
      }
      if (dh116_bus_cycle(&bus, f)) return 2;
      tgt_log[n] = sent[joint];
      pos_log[n++] = bus.fb.pos[joint];
   }

   if ((out = fopen(csv, "w")))
   {
      fprintf(out, "t_s,target,measured\n");
      for (i = 0; i < n; ++i) fprintf(out, "%.3f,%d,%d\n", i * 0.005, tgt_log[i], pos_log[i]);
      fclose(out);
   }
   for (j = 0; j <= 100 && j < n / 2; ++j) /* lag in cycles that best aligns measured with target */
   {
      double s = 0;
      for (i = j; i < n; ++i) s += (double)(tgt_log[i - j] - pos_log[i]) * (tgt_log[i - j] - pos_log[i]);
      s = sqrt(s / (n - j));
      if (s < best_rms) { best_rms = s; best = j; }
   }
   printf("RESULT: lag %d ms, RMS error after lag %.0f raw (%.1f%% of amp), %d samples -> %s\n",
          best * 5, best_rms, 100.0 * best_rms / amp, n, csv);

   g_stop = 0;
   printf("opening the hand ...\n");
   if (dh116_bus_move_wait(&bus, zero, vel, cur, 400)) return 2;
   dh116_bus_close(&bus);
   return 0;
}
