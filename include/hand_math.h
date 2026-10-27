/* hand_math.h - hand landmarks -> teleop features:
 * 21 MediaPipe-style world landmarks -> six features -> calibrated closure 0..1 -> One-Euro filter.
 * Palm rotation is left out: the physical hand has no wrist. Pure C, no SOEM.
 * Feature/actuator order: thumb sweep, thumb bend, index, middle, ring, little.
 */
#ifndef HAND_MATH_H
#define HAND_MATH_H
#include <stddef.h>

#define HM_N 6
#define HM_SAMPLES 60
#define HM_CAL_SECONDS 0.5

typedef struct
{
   double open[HM_N], closed[HM_N];
} hm_calib;

typedef struct
{
   double min_cutoff, beta, dcutoff;
   int has;
   double value[HM_N], speed[HM_N], time;
} hm_euro;

typedef struct
{
   hm_calib cal;
   char cal_path[512];
   hm_euro filt;
   double sample_t[HM_SAMPLES], sample_f[HM_SAMPLES][HM_N];
   int nsamples, head;
   char msg[64];
} hm_tracker;

/* 0 = ok, -1 = degenerate landmarks */
int hm_features(const float pts[21][3], int mirror, double out[HM_N]);

void hm_calib_default(hm_calib *c);
int hm_calib_validate(const hm_calib *c, char *err, size_t n);
/* missing file -> defaults; returns 0 ok, -1 bad file (err filled) */
int hm_calib_load(hm_calib *c, const char *path, char *err, size_t n);
int hm_calib_save(const hm_calib *c, const char *path);

void hm_euro_init(hm_euro *f, double min_cutoff, double beta);
void hm_euro_reset(hm_euro *f);
void hm_euro_apply(hm_euro *f, const double in[HM_N], double t, double out[HM_N]);

void hm_tracker_init(hm_tracker *t, const hm_calib *c, const char *path, double min_cutoff, double beta);
/* pts NULL = no hand. Returns 1 and fills closure when a hand was usable, else 0 (filters reset). */
int hm_tracker_update(hm_tracker *t, const float (*pts)[3], int mirror, double time_s, double closure[HM_N]);
/* kind: 1 open, 2 fist, 3 reset (TP_CAL_*). Result text in t->msg. */
void hm_tracker_calibrate(hm_tracker *t, int kind, double time_s);

#endif
