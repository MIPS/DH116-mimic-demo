/* teleop_proto.h - the only contract between hand_cam.py (camera, MediaPipe, GUI) and
 * dh116_teleop (math + EtherCAT). Fixed-size little-endian UDP datagrams on 127.0.0.1.
 * hand_cam.py mirrors these layouts with struct format strings; keep both in sync
 * (test/parity_check.py checks the sizes). Fields are naturally aligned, so no packing is needed.
 */
#ifndef TELEOP_PROTO_H
#define TELEOP_PROTO_H
#include <stdint.h>

#define TP_PORT 5006
#define TP_VERSION 1
#define TP_MAGIC_LANDMARKS 0x4D4C4844u /* "DHLM" */
#define TP_MAGIC_STATUS 0x54534844u    /* "DHST" */

/* tp_landmarks.flags */
#define TP_FLAG_DETECTED 0x01
#define TP_FLAG_QUIT 0x02 /* GUI closed: open the hand, disable, exit */

/* tp_landmarks.cal_cmd */
enum { TP_CAL_NONE = 0, TP_CAL_OPEN = 1, TP_CAL_FIST = 2, TP_CAL_RESET = 3 };

/* hand_cam.py -> dh116_teleop, one per camera frame. Python: "<IHBBIB3xd63f4x" */
typedef struct
{
   uint32_t magic;
   uint16_t version;
   uint8_t flags;
   uint8_t cal_cmd;
   uint32_t seq;
   uint8_t right_hand; /* 1 = landmarks are of a right hand; C mirrors them onto the left robot hand */
   uint8_t pad[3];
   double time_s;      /* capture time, monotonic seconds (drives the One-Euro filter) */
   float world[21][3]; /* MediaPipe hand_world_landmarks, metres */
   uint32_t pad2;      /* explicit tail padding (the double makes the struct 8-byte aligned) */
} tp_landmarks;
_Static_assert(sizeof(tp_landmarks) == 280, "tp_landmarks layout");

/* tp_status.state */
enum { TP_STATE_DRYRUN = 0, TP_STATE_RUNNING = 1, TP_STATE_STOPPING = 2 };

/* dh116_teleop -> hand_cam.py, one reply per landmarks packet. Python: "<IHBBI6f6h6hB3x64s"
 * Per-joint arrays are in actuator order: thumb sweep, thumb bend, index, middle, ring, little. */
typedef struct
{
   uint32_t magic;
   uint16_t version;
   uint8_t state;
   uint8_t valid; /* 1 = closure comes from a hand seen in this packet */
   uint32_t seq_echo;
   float closure[6];    /* filtered 0 (open) .. 1 (fist) */
   int16_t target[6];   /* raw 0..10000 last sent to the hand */
   int16_t measured[6]; /* raw position feedback from the hand */
   uint8_t have_fb;
   uint8_t pad[3];
   char msg[64]; /* calibration / status text, NUL-terminated */
} tp_status;
_Static_assert(sizeof(tp_status) == 128, "tp_status layout");

#endif
