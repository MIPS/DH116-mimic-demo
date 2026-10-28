/* hand_math.h - hand landmarks -> how closed each robot joint should be.
 * 21 MediaPipe-style world landmarks -> six joint features -> calibrated closure 0 (open) .. 1 (fist)
 * -> One-Euro filter. Palm rotation is left out: the robot hand has no wrist. Plain C, no SOEM.
 * Joint order: thumb sweep, thumb bend, index, middle, ring, little.
 */
#ifndef HAND_MATH_H
#define HAND_MATH_H
#include <stddef.h>

#define HM_JOINTS 6
#define HM_RECENT 60 /* recent feature samples kept for calibration (2 s at 30 fps) */

/* hm_tracker_calibrate kinds */
#define HM_CAL_OPEN 1
#define HM_CAL_FIST 2
#define HM_CAL_RESET 3

/* Feature value of an open hand and of a fist, per joint. */
typedef struct
{
    double open[HM_JOINTS], closed[HM_JOINTS];
} hm_calib;

typedef struct
{
    double min_cutoff, beta;
    int started;
    double value[HM_JOINTS], speed[HM_JOINTS], time;
} hm_euro;

typedef struct
{
    hm_calib cal;
    char cal_path[512];
    hm_euro filter;
    double recent_time[HM_RECENT], recent_feat[HM_RECENT][HM_JOINTS]; /* ring buffer */
    int n_recent, next_slot;
    char msg[64]; /* calibration status for the GUI */
} hm_tracker;

/* Loads the calibration from cal_path (no file = defaults). Returns 0, or -1 with err filled for a bad file. */
int hm_tracker_init(hm_tracker *t, const char *cal_path, double min_cutoff, double beta, char *err, size_t n);

/* pts NULL = no hand. Returns 1 and fills closure[] when the hand was usable; else 0 and the filter restarts. */
int hm_tracker_update(hm_tracker *t, const float (*pts)[3], int mirror, double time_s, double closure[HM_JOINTS]);

/* Store the last 0.5 s of features as open hand / fist, or reset to defaults. Result text in t->msg. */
void hm_tracker_calibrate(hm_tracker *t, int kind, double time_s);

#endif
