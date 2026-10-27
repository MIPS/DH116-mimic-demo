/* hand_math.c - see hand_math.h. */
#include "hand_math.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define RAD(d) ((d) * M_PI / 180.0)

static const char *LABELS[HM_N] = {"Thumb sweep", "Thumb bend", "Index", "Middle", "Ring", "Little"};
static const int THUMB[4] = {1, 2, 3, 4};
static const int FINGERS[4][4] = {{5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}, {17, 18, 19, 20}};

/* ---------------------------------------------------------------- vectors */

typedef struct { double x, y, z; } v3;

static v3 sub(v3 a, v3 b) { return (v3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static v3 scale(v3 a, double s) { return (v3){a.x * s, a.y * s, a.z * s}; }
static double dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static v3 cross(v3 a, v3 b) { return (v3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static double norm(v3 a) { return sqrt(dot(a, a)); }

static int unit(v3 a, v3 *out)
{
   double n = norm(a);
   if (!isfinite(n) || n < 1e-9) return -1;
   *out = scale(a, 1.0 / n);
   return 0;
}

static int angle_between(v3 a, v3 b, double *out)
{
   if (unit(a, &a) || unit(b, &b)) return -1;
   *out = atan2(norm(cross(a, b)), dot(a, b));
   return 0;
}

/* ---------------------------------------------------------------- features */

int hm_features(const float pts[21][3], int mirror, double out[HM_N])
{
   v3 p[21], along, across, normal;
   double width, a, b, c;
   int i;
   for (i = 0; i < 21; ++i)
   {
      p[i] = (v3){mirror ? -(double)pts[i][0] : (double)pts[i][0], pts[i][1], pts[i][2]};
      if (!isfinite(p[i].x) || !isfinite(p[i].y) || !isfinite(p[i].z)) return -1;
   }
   /* palm_frame: columns (palm normal, across, along) */
   if (unit(sub(p[9], p[0]), &along)) return -1;
   across = sub(p[17], p[5]);
   if (unit(sub(across, scale(along, dot(across, along))), &across)) return -1;
   normal = cross(across, along);

   width = norm(sub(p[17], p[5]));
   out[0] = dot(sub(p[THUMB[3]], p[5]), across) / width;
   if (angle_between(sub(p[THUMB[1]], p[THUMB[0]]), sub(p[THUMB[2]], p[THUMB[1]]), &a)) return -1;
   if (angle_between(sub(p[THUMB[2]], p[THUMB[1]]), sub(p[THUMB[3]], p[THUMB[2]]), &b)) return -1;
   out[1] = a + b;
   for (i = 0; i < 4; ++i)
   {
      const int *f = FINGERS[i];
      v3 proximal = sub(p[f[1]], p[f[0]]), intermediate = sub(p[f[2]], p[f[1]]);
      double flex = atan2(dot(proximal, normal), dot(proximal, along));
      if (angle_between(proximal, intermediate, &b)) return -1;
      if (angle_between(intermediate, sub(p[f[3]], p[f[2]]), &c)) return -1;
      out[2 + i] = flex + b + c;
   }
   for (i = 0; i < HM_N; ++i)
      if (!isfinite(out[i])) return -1;
   return 0;
}

/* ---------------------------------------------------------------- calibration */

void hm_calib_default(hm_calib *c)
{
   int i;
   c->open[0] = -0.9; c->closed[0] = 0.6;
   c->open[1] = RAD(15); c->closed[1] = RAD(70);
   for (i = 2; i < HM_N; ++i) { c->open[i] = RAD(30); c->closed[i] = RAD(200); }
}

int hm_calib_validate(const hm_calib *c, char *err, size_t n)
{
   static const double min_span[HM_N] = {0.3, RAD(15), RAD(45), RAD(45), RAD(45), RAD(45)};
   int i;
   for (i = 0; i < HM_N; ++i)
      if (!isfinite(c->open[i]) || !isfinite(c->closed[i])) { snprintf(err, n, "values must be finite"); return -1; }
   for (i = 0; i < HM_N; ++i)
      if (c->closed[i] - c->open[i] < min_span[i]) { snprintf(err, n, "%s: fist not more closed than open", LABELS[i]); return -1; }
   return 0;
}

/* Read the n numbers after "key": in a JSON text, ignoring brackets/commas (nested arrays flatten). */
static int json_numbers(const char *text, const char *key, double *out, int n)
{
   char pat[64];
   const char *p;
   int i;
   snprintf(pat, sizeof(pat), "\"%s\"", key);
   if (!(p = strstr(text, pat)) || !(p = strchr(p + strlen(pat), ':'))) return -1;
   ++p;
   for (i = 0; i < n; ++i)
   {
      char *end;
      while (*p && strchr(" \t\r\n[],", *p)) ++p;
      out[i] = strtod(p, &end);
      if (end == p) return -1;
      p = end;
   }
   return 0;
}

int hm_calib_load(hm_calib *c, const char *path, char *err, size_t n)
{
   char buf[8192];
   double version = 0;
   size_t len;
   FILE *f = fopen(path, "r");
   hm_calib_default(c);
   if (!f) return 0;
   len = fread(buf, 1, sizeof(buf) - 1, f);
   fclose(f);
   buf[len] = 0;
   if (!strstr(buf, "\"dh116-hand-teleop\"") || json_numbers(buf, "version", &version, 1) || version != 1.0 ||
       json_numbers(buf, "open", c->open, HM_N) || json_numbers(buf, "closed", c->closed, HM_N))
   {
      snprintf(err, n, "%s is not a version 1 dh116-hand-teleop calibration", path);
      return -1;
   }
   return hm_calib_validate(c, err, n);
}

int hm_calib_save(const hm_calib *c, const char *path)
{
   char tmp[600];
   const char *uid = getenv("SUDO_UID"), *gid = getenv("SUDO_GID");
   FILE *f;
   int i;
   snprintf(tmp, sizeof(tmp), "%s.part", path);
   if (!(f = fopen(tmp, "w"))) return -1;
   fprintf(f, "{\n  \"protocol\": \"dh116-hand-teleop\",\n  \"version\": 1,\n  \"features\": [");
   for (i = 0; i < HM_N; ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", LABELS[i]);
   fprintf(f, "],\n  \"open\": [");
   for (i = 0; i < HM_N; ++i) fprintf(f, "%s%.17g", i ? ", " : "", c->open[i]);
   fprintf(f, "],\n  \"closed\": [");
   for (i = 0; i < HM_N; ++i) fprintf(f, "%s%.17g", i ? ", " : "", c->closed[i]);
   fprintf(f, "]\n}\n");
   if (fclose(f) != 0) return -1;
   if (uid && gid && chown(tmp, (uid_t)atoi(uid), (gid_t)atoi(gid))) { /* best effort when run with sudo */ }
   return rename(tmp, path);
}

/* ---------------------------------------------------------------- One-Euro filter */

void hm_euro_init(hm_euro *f, double min_cutoff, double beta)
{
   f->min_cutoff = min_cutoff;
   f->beta = beta;
   f->dcutoff = 1.0;
   hm_euro_reset(f);
}

void hm_euro_reset(hm_euro *f) { f->has = 0; }

static double smoothing(double cutoff, double elapsed) { return 1.0 / (1.0 + 1.0 / (2.0 * M_PI * cutoff * elapsed)); }

void hm_euro_apply(hm_euro *f, const double in[HM_N], double t, double out[HM_N])
{
   double elapsed;
   int i;
   if (!f->has)
   {
      memcpy(f->value, in, sizeof(f->value));
      memset(f->speed, 0, sizeof(f->speed));
      f->time = t;
      f->has = 1;
      memcpy(out, in, sizeof(f->value));
      return;
   }
   elapsed = t - f->time;
   if (elapsed <= 0) { memcpy(out, f->value, sizeof(f->value)); return; }
   for (i = 0; i < HM_N; ++i)
   {
      double speed = (in[i] - f->value[i]) / elapsed, cutoff;
      f->speed[i] += smoothing(f->dcutoff, elapsed) * (speed - f->speed[i]);
      cutoff = f->min_cutoff + f->beta * fabs(f->speed[i]);
      f->value[i] += smoothing(cutoff, elapsed) * (in[i] - f->value[i]);
   }
   f->time = t;
   memcpy(out, f->value, sizeof(f->value));
}

/* ---------------------------------------------------------------- tracker */

static double clip01(double v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

void hm_tracker_init(hm_tracker *t, const hm_calib *c, const char *path, double min_cutoff, double beta)
{
   memset(t, 0, sizeof(*t));
   t->cal = *c;
   snprintf(t->cal_path, sizeof(t->cal_path), "%s", path ? path : "");
   hm_euro_init(&t->filt, min_cutoff, beta);
   snprintf(t->msg, sizeof(t->msg), "%s", path && access(path, F_OK) == 0 ? "Using saved calibration" : "Default calibration");
}

int hm_tracker_update(hm_tracker *t, const float (*pts)[3], int mirror, double time_s, double closure[HM_N])
{
   double f[HM_N], raw[HM_N];
   int i;
   if (!pts || hm_features(pts, mirror, f))
   {
      hm_euro_reset(&t->filt);
      return 0;
   }
   t->sample_t[t->head] = time_s;
   memcpy(t->sample_f[t->head], f, sizeof(f));
   t->head = (t->head + 1) % HM_SAMPLES;
   if (t->nsamples < HM_SAMPLES) ++t->nsamples;
   for (i = 0; i < HM_N; ++i) raw[i] = clip01((f[i] - t->cal.open[i]) / (t->cal.closed[i] - t->cal.open[i]));
   hm_euro_apply(&t->filt, raw, time_s, closure);
   for (i = 0; i < HM_N; ++i) closure[i] = clip01(closure[i]);
   return 1;
}

static int cmp_double(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return (x > y) - (x < y);
}

void hm_tracker_calibrate(hm_tracker *t, int kind, double time_s)
{
   hm_calib c;
   char err[48];
   int i, k, n = 0;
   if (kind == 3) hm_calib_default(&c);
   else
   {
      double col[HM_SAMPLES], med[HM_N];
      for (i = 0; i < HM_N; ++i)
      {
         n = 0;
         for (k = 0; k < t->nsamples; ++k)
            if (time_s - t->sample_t[k] <= HM_CAL_SECONDS) col[n++] = t->sample_f[k][i];
         if (n < 5) { snprintf(t->msg, sizeof(t->msg), "Unchanged: hold the pose in view, press again"); return; }
         qsort(col, n, sizeof(double), cmp_double);
         med[i] = n % 2 ? col[n / 2] : 0.5 * (col[n / 2 - 1] + col[n / 2]);
      }
      c = t->cal;
      memcpy(kind == 1 ? c.open : c.closed, med, sizeof(med));
      if (hm_calib_validate(&c, err, sizeof(err))) { snprintf(t->msg, sizeof(t->msg), "Unchanged: %s", err); return; }
   }
   if (t->cal_path[0] && hm_calib_save(&c, t->cal_path)) { snprintf(t->msg, sizeof(t->msg), "Unchanged: cannot write calibration"); return; }
   t->cal = c;
   hm_euro_reset(&t->filt);
   snprintf(t->msg, sizeof(t->msg), "%s", kind == 1 ? "Saved open hand" : kind == 2 ? "Saved fist" : "Restored default calibration");
}
