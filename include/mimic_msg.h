/* mimic_msg.h - the UDP messages between hand_cam (camera + neural nets) and mimic_ctrl (math + EtherCAT).
 * Both programs run on the same machine and talk over 127.0.0.1, so the structs are sent as raw bytes.
 * A receiver drops any packet whose size or version does not match. Used from C and C++.
 */
#ifndef MIMIC_MSG_H
#define MIMIC_MSG_H
#include <assert.h>
#include <stdint.h>

#define MSG_PORT 5006
#define MSG_VERSION 2

/* landmarks_msg.flags */
#define MSG_HAND_SEEN 0x01
#define MSG_QUIT 0x02       /* GUI closed: open the hand, disable the joints, exit */
#define MSG_RIGHT_HAND 0x04 /* landmarks are of a right hand; they get mirrored onto the left robot hand */

/* landmarks_msg.calibrate */
enum { MSG_CAL_NONE = 0, MSG_CAL_OPEN = 1, MSG_CAL_FIST = 2, MSG_CAL_RESET = 3 };

/* hand_cam -> mimic_ctrl, one per camera frame. */
typedef struct
{
    uint16_t version;
    uint8_t flags;
    uint8_t calibrate;
    float world[21][3]; /* hand landmarks in metres (the landmark net's "world" output) */
    double time_s;      /* capture time, monotonic seconds; drives the One-Euro filter */
} landmarks_msg;
static_assert(sizeof(landmarks_msg) == 264, "landmarks_msg layout");

/* mimic_ctrl -> hand_cam, one reply per landmarks_msg.
 * Joint order: thumb sweep, thumb bend, index, middle, ring, little. */
typedef struct
{
    uint16_t version;
    uint8_t dry_run;
    uint8_t got_feedback;
    int16_t target[6];   /* raw 0..10000 sent to the hand */
    int16_t measured[6]; /* raw position feedback from the hand */
    char text[64];       /* calibration result or other status text */
} status_msg;
static_assert(sizeof(status_msg) == 92, "status_msg layout");

#endif
