/*
 * dh116_teleop.c - webcam teleop of the physical DH116 hand.
 *
 *   sudo ./build/dh116_teleop <ifname> [options]     drive the hand
 *        ./build/dh116_teleop --dry-run [options]    same pipeline, no EtherCAT (for the GUI / tests)
 *
 * hand_cam sends hand world landmarks (teleop_proto.h) to 127.0.0.1:<port>. This program
 * turns them into closures (hand_math.c), maps closure 0..1 to raw 0..10000 per joint and streams
 * them to the hand: a data frame with the new targets, then one MOVE control frame, then data
 * frames again, repeated whenever a target moves past --deadband.
 * One thread: every 5 ms cycle drains the UDP socket without blocking, then exchanges process data.
 *
 * Hand lost / GUI silent: the last pose is held. Quit (GUI Q/Esc, Ctrl+C, SIGTERM): the hand
 * opens, joints are disabled, bus goes to INIT. Working counter fault: bus to INIT, exit 2.
 *
 * Options: --port N (5006)  --cur 50..1000 (1000)  --vel 100..20000 (20000)
 *          --deadband RAW 0..2000 (50)  --pre N data cycles before each MOVE, 1..20 (1)
 *          --calibration FILE  --min-cutoff HZ (1.0)  --beta B (1.0)  --quiet
 */
#include <arpa/inet.h>
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

#include "dh116_bus.h"
#include "hand_math.h"
#include "teleop_proto.h"

#ifndef TELEOP_DIR
#define TELEOP_DIR "."
#endif

/* actuator order (thumb sweep, thumb bend, index, middle, ring, little) -> DH116 joint 1..6 */
static const int ACT_JOINT[HM_N] = {1, 2, 6, 5, 4, 3};

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int s) { (void)s; g_stop = 1; }

typedef struct
{
   int sock, quit, have_seq, valid, verbose;
   uint32_t last_seq;
   hm_tracker tracker;
   double closure[HM_N];
   int desired[7], have_desired;
   unsigned packets, detected, dropped;
} teleop;

static int clamp_raw(int v) { return v < 0 ? 0 : (v > 10000 ? 10000 : v); }

static void reply(teleop *t, const struct sockaddr_in *to, uint32_t seq, int state, const int sent[7], const dh116_bus *bus)
{
   tp_status s;
   int i;
   memset(&s, 0, sizeof(s));
   s.magic = TP_MAGIC_STATUS;
   s.version = TP_VERSION;
   s.state = (uint8_t)state;
   s.valid = (uint8_t)t->valid;
   s.seq_echo = seq;
   for (i = 0; i < HM_N; ++i)
   {
      s.closure[i] = (float)t->closure[i];
      s.target[i] = (int16_t)sent[ACT_JOINT[i]];
      s.measured[i] = bus ? (int16_t)bus->fb.pos[ACT_JOINT[i]] : 0;
   }
   s.have_fb = (uint8_t)(bus && bus->have_fb);
   snprintf(s.msg, sizeof(s.msg), "%s", t->tracker.msg);
   sendto(t->sock, &s, sizeof(s), 0, (const struct sockaddr *)to, sizeof(*to));
}

/* drain every waiting datagram; the newest valid one wins */
static void poll_udp(teleop *t, int state, const int sent[7], const dh116_bus *bus)
{
   tp_landmarks m;
   struct sockaddr_in from;
   socklen_t fl;
   ssize_t n;
   int i;
   for (;;)
   {
      fl = sizeof(from);
      n = recvfrom(t->sock, &m, sizeof(m), MSG_DONTWAIT | MSG_TRUNC, (struct sockaddr *)&from, &fl);
      if (n < 0) return;
      if (n != (ssize_t)sizeof(m) || m.magic != TP_MAGIC_LANDMARKS || m.version != TP_VERSION || !isfinite(m.time_s) ||
          m.cal_cmd > TP_CAL_RESET)
      { t->dropped++; continue; }
      if (m.flags & TP_FLAG_QUIT) { t->quit = 1; continue; } /* honoured even from a restarted GUI */
      if (t->have_seq && m.seq <= t->last_seq && m.seq != 0) /* seq 0 = a (re)started GUI */
      { t->dropped++; continue; }
      t->have_seq = 1;
      t->last_seq = m.seq;
      t->packets++;
      t->valid = hm_tracker_update(&t->tracker, (m.flags & TP_FLAG_DETECTED) ? (const float (*)[3])m.world : NULL,
                                   m.right_hand, m.time_s, t->closure);
      if (t->valid)
      {
         t->detected++;
         for (i = 0; i < HM_N; ++i) t->desired[ACT_JOINT[i]] = clamp_raw((int)lround(t->closure[i] * 10000.0));
         t->have_desired = 1;
      }
      if (m.cal_cmd != TP_CAL_NONE)
      {
         hm_tracker_calibrate(&t->tracker, m.cal_cmd, m.time_s);
         printf("%s\n", t->tracker.msg);
         fflush(stdout);
      }
      reply(t, &from, m.seq, state, sent, bus);
   }
}

static void pace(struct timespec *next)
{
   next->tv_nsec += DH116_CYCLE_US * 1000L;
   while (next->tv_nsec >= 1000000000L) { next->tv_nsec -= 1000000000L; next->tv_sec++; }
   clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, next, NULL);
}

static void usage(const char *p)
{
   printf("usage: sudo %s <ifname> | %s --dry-run   [--port N] [--cur 50..1000] [--vel 100..20000]\n"
          "       [--deadband 0..2000] [--pre 1..20] [--calibration FILE] [--min-cutoff HZ] [--beta B] [--quiet]\n", p, p);
}

int main(int argc, char **argv)
{
   static dh116_bus bus;
   static teleop t;
   const char *ifname = NULL, *cal_path = TELEOP_DIR "/hand_calibration.json";
   int dry = 0, port = TP_PORT, cur = 1000, vel = 20000, deadband = 50, pre = 1, i, j;
   double min_cutoff = 1.0, beta = 1.0;
   int sent[7] = {0}, vv[7], cc[7], phase = 0, pre_left = 0; /* phase 0 idle, 1 pre-data, 2 MOVE */
   uint16_t mv[7];
   uint8_t f[DH116_RPDO_BYTES];
   hm_calib cal;
   char err[160];
   struct sockaddr_in addr;
   struct timespec next, last_log;
   unsigned cycles = 0, moves = 0;

   t.verbose = 1;
   for (i = 1; i < argc; ++i)
   {
      const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
      if (!strcmp(a, "--dry-run")) dry = 1;
      else if (!strcmp(a, "--quiet")) t.verbose = 0;
      else if (!strcmp(a, "--port") && v) port = atoi(argv[++i]);
      else if (!strcmp(a, "--cur") && v) cur = atoi(argv[++i]);
      else if (!strcmp(a, "--vel") && v) vel = atoi(argv[++i]);
      else if (!strcmp(a, "--deadband") && v) deadband = atoi(argv[++i]);
      else if (!strcmp(a, "--pre") && v) pre = atoi(argv[++i]);
      else if (!strcmp(a, "--calibration") && v) cal_path = argv[++i];
      else if (!strcmp(a, "--min-cutoff") && v) min_cutoff = atof(argv[++i]);
      else if (!strcmp(a, "--beta") && v) beta = atof(argv[++i]);
      else if (a[0] != '-' && !ifname) ifname = a;
      else { usage(argv[0]); return 1; }
   }
   if ((!dry) == (!ifname) || port < 1 || port > 65535 || cur < 50 || cur > 1000 || vel < 100 || vel > 20000 ||
       deadband < 0 || deadband > 2000 || pre < 1 || pre > 20 || !(min_cutoff > 0) || !(beta >= 0))
   { usage(argv[0]); return 1; }

   if (hm_calib_load(&cal, cal_path, err, sizeof(err))) { fprintf(stderr, "calibration: %s\n", err); return 1; }
   hm_tracker_init(&t.tracker, &cal, cal_path, min_cutoff, beta);

   t.sock = socket(AF_INET, SOCK_DGRAM, 0);
   memset(&addr, 0, sizeof(addr));
   addr.sin_family = AF_INET;
   addr.sin_port = htons((uint16_t)port);
   addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); /* local only */
   if (t.sock < 0 || bind(t.sock, (struct sockaddr *)&addr, sizeof(addr)))
   { fprintf(stderr, "cannot listen on 127.0.0.1:%d: %s\n", port, strerror(errno)); return 1; }

   signal(SIGINT, on_signal);
   signal(SIGTERM, on_signal);
   for (j = 0; j < 7; ++j) { vv[j] = vel; cc[j] = cur; mv[j] = DH116_CTRL_MOVE; }

   if (!dry)
   {
      if (dh116_bus_open(&bus, ifname)) return 2;
      if (!bus.have_fb) { printf("no joint feedback, refusing to stream targets\n"); dh116_bus_close(&bus); return 2; }
      for (j = 1; j <= 6; ++j) sent[j] = clamp_raw(bus.fb.pos[j]); /* start by holding the current pose */
   }
   memcpy(t.desired, sent, sizeof(sent));
   printf("teleop %s: listening on 127.0.0.1:%d, vel %d, cur %d, deadband %d, calibration %s (%s)\n",
          dry ? "DRY RUN (no EtherCAT)" : ifname, port, vel, cur, deadband, cal_path, t.tracker.msg);
   fflush(stdout);

   clock_gettime(CLOCK_MONOTONIC, &next);
   last_log = next;
   while (!g_stop && !t.quit)
   {
      poll_udp(&t, dry ? TP_STATE_DRYRUN : TP_STATE_RUNNING, sent, dry ? NULL : &bus);

      if (phase == 0 && t.have_desired)
      {
         int change = 0;
         for (j = 1; j <= 6; ++j)
            if (abs(t.desired[j] - sent[j]) > deadband) change = 1;
         if (change) { memcpy(sent, t.desired, sizeof(sent)); phase = 1; pre_left = pre; moves++; }
      }
      if (phase == 2) { dh116_frame_ctrl(f, mv); phase = 0; }
      else
      {
         dh116_frame_data(f, sent, vv, cc);
         if (phase == 1 && --pre_left == 0) phase = 2;
      }

      if (dry) pace(&next);
      else if (dh116_bus_cycle(&bus, f)) return 2;

      if (t.verbose && ++cycles % 200 == 0) /* about once per second */
      {
         struct timespec now;
         double dt;
         clock_gettime(CLOCK_MONOTONIC, &now);
         dt = (now.tv_sec - last_log.tv_sec) + (now.tv_nsec - last_log.tv_nsec) * 1e-9;
         last_log = now;
         printf("%5.1f pkt/s %5.1f det/s %4.1f moves/s drop %u | target j1..6:", t.packets / dt, t.detected / dt, moves / dt, t.dropped);
         for (j = 1; j <= 6; ++j) printf(" %5d", sent[j]);
         if (!dry && bus.have_fb) { printf(" | pos:"); for (j = 1; j <= 6; ++j) printf(" %5d", bus.fb.pos[j]); }
         printf("\n");
         fflush(stdout);
         t.packets = t.detected = moves = 0;
      }
   }

   printf("stopping: %s\n", t.quit ? "GUI quit" : "signal");
   if (!dry)
   {
      int zero[7] = {0};
      signal(SIGINT, SIG_IGN); /* finish opening + disabling even if Ctrl+C is pressed again */
      printf("opening the hand ...\n");
      if (dh116_bus_move_wait(&bus, zero, vel, cur, 400)) return 2;
      dh116_bus_close(&bus);
   }
   close(t.sock);
   return 0;
}
