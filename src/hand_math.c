/* hand_math.c - see hand_math.h. */
#include "hand_math.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define RAD(d) ((d) * M_PI / 180.0)
#define CAL_SECONDS 0.5   /* calibration uses the features of the last half second */
#define EURO_DCUTOFF 1.0  /* Hz, One-Euro cutoff for the speed estimate */

/* MediaPipe landmark numbers. Each finger has 4 in a row: knuckle (MCP), PIP, DIP, tip. */
enum { WRIST = 0, THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4,
       INDEX_MCP = 5, MIDDLE_MCP = 9, RING_MCP = 13, PINKY_MCP = 17 };

static const char *LABELS[HM_JOINTS] = {"Thumb sweep", "Thumb bend", "Index", "Middle", "Ring", "Little"};

/* ---------------------------------------------------------------- vectors */

typedef struct { double x, y, z; } v3;

static v3 sub(v3 a, v3 b) { return (v3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static v3 scale(v3 a, double s) { return (v3){a.x * s, a.y * s, a.z * s}; }
static double dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static v3 cross(v3 a, v3 b) { return (v3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static double norm(v3 a) { return sqrt(dot(a, a)); }
static double clip01(double v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

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

/* Six joint features of one hand: radians, except the thumb sweep (palm widths). -1 = degenerate landmarks. */
static int features(const float pts[21][3], int mirror, double out[HM_JOINTS])
{
    static const int FINGER_MCP[4] = {INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP};
    v3 p[21], along, across, normal;
    double width, a, b;
    for (int i = 0; i < 21; ++i)
    {
        p[i] = (v3){mirror ? -(double)pts[i][0] : (double)pts[i][0], pts[i][1], pts[i][2]};
        if (!isfinite(p[i].x) || !isfinite(p[i].y) || !isfinite(p[i].z)) return -1;
    }
    /* Palm axes: along = wrist to middle knuckle, across = index to little knuckle, normal = out of the palm. */
    if (unit(sub(p[MIDDLE_MCP], p[WRIST]), &along)) return -1;
    across = sub(p[PINKY_MCP], p[INDEX_MCP]);
    width = norm(across);
    if (unit(sub(across, scale(along, dot(across, along))), &across)) return -1;
    normal = cross(across, along);

    /* Thumb sweep: how far the thumb tip is across the palm from the index knuckle. */
    out[0] = dot(sub(p[THUMB_TIP], p[INDEX_MCP]), across) / width;
    /* Thumb bend: bend at the MCP joint + bend at the IP joint. */
    if (angle_between(sub(p[THUMB_MCP], p[THUMB_CMC]), sub(p[THUMB_IP], p[THUMB_MCP]), &a)) return -1;
    if (angle_between(sub(p[THUMB_IP], p[THUMB_MCP]), sub(p[THUMB_TIP], p[THUMB_IP]), &b)) return -1;
    out[1] = a + b;
    /* Fingers: knuckle flex out of the palm plane + bend at the two finger joints. */
    for (int i = 0; i < 4; ++i)
    {
        int m = FINGER_MCP[i];
        v3 proximal = sub(p[m + 1], p[m]), middle = sub(p[m + 2], p[m + 1]), distal = sub(p[m + 3], p[m + 2]);
        double flex = atan2(dot(proximal, normal), dot(proximal, along));
        if (angle_between(proximal, middle, &a) || angle_between(middle, distal, &b)) return -1;
        out[2 + i] = flex + a + b;
    }
    for (int i = 0; i < HM_JOINTS; ++i)
        if (!isfinite(out[i])) return -1;
    return 0;
}

/* ---------------------------------------------------------------- calibration file */

static void calib_default(hm_calib *c)
{
    c->open[0] = -0.9;
    c->closed[0] = 0.6;
    c->open[1] = RAD(15);
    c->closed[1] = RAD(70);
    for (int i = 2; i < HM_JOINTS; ++i)
    {
        c->open[i] = RAD(30);
        c->closed[i] = RAD(200);
    }
}

static int calib_validate(const hm_calib *c, char *err, size_t n)
{
    static const double min_span[HM_JOINTS] = {0.3, RAD(15), RAD(45), RAD(45), RAD(45), RAD(45)};
    for (int i = 0; i < HM_JOINTS; ++i)
        if (!isfinite(c->open[i]) || !isfinite(c->closed[i]))
        {
            snprintf(err, n, "values must be finite");
            return -1;
        }
    for (int i = 0; i < HM_JOINTS; ++i)
        if (c->closed[i] - c->open[i] < min_span[i])
        {
            snprintf(err, n, "%s: fist not more closed than open", LABELS[i]);
            return -1;
        }
    return 0;
}

/* Read the n numbers after "key": in a JSON text, ignoring brackets and commas. */
static int json_numbers(const char *text, const char *key, double *out, int n)
{
    char pat[64];
    const char *p;
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    if (!(p = strstr(text, pat)) || !(p = strchr(p + strlen(pat), ':'))) return -1;
    ++p;
    for (int i = 0; i < n; ++i)
    {
        char *end;
        while (*p && strchr(" \t\r\n[],", *p)) ++p;
        out[i] = strtod(p, &end);
        if (end == p) return -1;
        p = end;
    }
    return 0;
}

/* The "dh116-hand-teleop" name is kept so existing calibration files still load. */
#define CAL_FILE_TAG "dh116-hand-teleop"

/* 1 = loaded, 0 = no file (defaults), -1 = bad file (err filled). */
static int calib_load(hm_calib *c, const char *path, char *err, size_t n)
{
    char buf[8192];
    double version = 0;
    size_t len;
    FILE *f = fopen(path, "r");
    calib_default(c);
    if (!f) return 0;
    len = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[len] = 0;
    if (!strstr(buf, "\"" CAL_FILE_TAG "\"") || json_numbers(buf, "version", &version, 1) || version != 1.0 ||
        json_numbers(buf, "open", c->open, HM_JOINTS) || json_numbers(buf, "closed", c->closed, HM_JOINTS))
    {
        snprintf(err, n, "%s is not a version 1 %s calibration", path, CAL_FILE_TAG);
        return -1;
    }
    return calib_validate(c, err, n) ? -1 : 1;
}

/* Write to a temporary file, then rename, so a crash never leaves half a file. */
static int calib_save(const hm_calib *c, const char *path)
{
    char tmp[600];
    const char *uid = getenv("SUDO_UID"), *gid = getenv("SUDO_GID");
    FILE *f;
    snprintf(tmp, sizeof(tmp), "%s.part", path);
    if (!(f = fopen(tmp, "w"))) return -1;
    fprintf(f, "{\n  \"protocol\": \"" CAL_FILE_TAG "\",\n  \"version\": 1,\n  \"features\": [");
    for (int i = 0; i < HM_JOINTS; ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", LABELS[i]);
    fprintf(f, "],\n  \"open\": [");
    for (int i = 0; i < HM_JOINTS; ++i) fprintf(f, "%s%.17g", i ? ", " : "", c->open[i]);
    fprintf(f, "],\n  \"closed\": [");
    for (int i = 0; i < HM_JOINTS; ++i) fprintf(f, "%s%.17g", i ? ", " : "", c->closed[i]);
    fprintf(f, "]\n}\n");
    if (fclose(f) != 0) return -1;
    /* Under sudo, give the file back to the user. Best effort: a failure is ignored. */
    if (uid && gid && chown(tmp, (uid_t)atoi(uid), (gid_t)atoi(gid))) { }
    return rename(tmp, path);
}

/* ---------------------------------------------------------------- One-Euro filter */

static double smoothing(double cutoff, double elapsed) { return 1.0 / (1.0 + 1.0 / (2.0 * M_PI * cutoff * elapsed)); }

static void euro_apply(hm_euro *f, const double in[HM_JOINTS], double t, double out[HM_JOINTS])
{
    double elapsed = t - f->time;
    if (!f->started)
    {
        memcpy(f->value, in, sizeof(f->value));
        memset(f->speed, 0, sizeof(f->speed));
        f->time = t;
        f->started = 1;
    }
    else if (elapsed > 0)
    {
        for (int i = 0; i < HM_JOINTS; ++i)
        {
            double speed = (in[i] - f->value[i]) / elapsed, cutoff;
            f->speed[i] += smoothing(EURO_DCUTOFF, elapsed) * (speed - f->speed[i]);
            cutoff = f->min_cutoff + f->beta * fabs(f->speed[i]);
            f->value[i] += smoothing(cutoff, elapsed) * (in[i] - f->value[i]);
        }
        f->time = t;
    }
    memcpy(out, f->value, sizeof(f->value));
}

/* ---------------------------------------------------------------- tracker */

int hm_tracker_init(hm_tracker *t, const char *cal_path, double min_cutoff, double beta, char *err, size_t n)
{
    int loaded;
    memset(t, 0, sizeof(*t));
    snprintf(t->cal_path, sizeof(t->cal_path), "%s", cal_path);
    t->filter.min_cutoff = min_cutoff;
    t->filter.beta = beta;
    loaded = calib_load(&t->cal, cal_path, err, n);
    snprintf(t->msg, sizeof(t->msg), "%s", loaded > 0 ? "Using saved calibration" : "Default calibration");
    return loaded < 0 ? -1 : 0;
}

int hm_tracker_update(hm_tracker *t, const float (*pts)[3], int mirror, double time_s, double closure[HM_JOINTS])
{
    double f[HM_JOINTS], raw[HM_JOINTS];
    if (!pts || features(pts, mirror, f))
    {
        t->filter.started = 0;
        return 0;
    }
    t->recent_time[t->next_slot] = time_s;
    memcpy(t->recent_feat[t->next_slot], f, sizeof(f));
    t->next_slot = (t->next_slot + 1) % HM_RECENT;
    if (t->n_recent < HM_RECENT) ++t->n_recent;

    for (int i = 0; i < HM_JOINTS; ++i) raw[i] = clip01((f[i] - t->cal.open[i]) / (t->cal.closed[i] - t->cal.open[i]));
    euro_apply(&t->filter, raw, time_s, closure); /* stays in 0..1: it only averages values in 0..1 */
    return 1;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

void hm_tracker_calibrate(hm_tracker *t, int kind, double time_s)
{
    hm_calib c = t->cal;
    double *pose = kind == HM_CAL_OPEN ? c.open : c.closed;
    char err[48];

    if (kind == HM_CAL_RESET)
    {
        /* No file means default calibration. */
        if (remove(t->cal_path) && errno != ENOENT)
        {
            snprintf(t->msg, sizeof(t->msg), "Unchanged: cannot delete calibration");
            return;
        }
        calib_default(&t->cal);
        t->filter.started = 0;
        snprintf(t->msg, sizeof(t->msg), "Restored default calibration");
        return;
    }
    /* The median of the last CAL_SECONDS of each feature becomes the new open / fist value. */
    for (int i = 0; i < HM_JOINTS; ++i)
    {
        double col[HM_RECENT];
        int n = 0;
        for (int k = 0; k < t->n_recent; ++k)
            if (time_s - t->recent_time[k] <= CAL_SECONDS) col[n++] = t->recent_feat[k][i];
        if (n < 5)
        {
            snprintf(t->msg, sizeof(t->msg), "Unchanged: hold the pose in view, press again");
            return;
        }
        qsort(col, n, sizeof(double), cmp_double);
        pose[i] = n % 2 ? col[n / 2] : 0.5 * (col[n / 2 - 1] + col[n / 2]);
    }
    if (calib_validate(&c, err, sizeof(err)))
    {
        snprintf(t->msg, sizeof(t->msg), "Unchanged: %s", err);
        return;
    }
    if (calib_save(&c, t->cal_path))
    {
        snprintf(t->msg, sizeof(t->msg), "Unchanged: cannot write calibration");
        return;
    }
    t->cal = c;
    t->filter.started = 0;
    snprintf(t->msg, sizeof(t->msg), "%s", kind == HM_CAL_OPEN ? "Saved open hand" : "Saved fist");
}
