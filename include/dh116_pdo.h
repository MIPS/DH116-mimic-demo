/* dh116_pdo.h - DH116EC process data, no Leadshine library needed.
 * Decoded from Leadshine's x86 library (no hand attached) and a capture; tested on a real hand
 * for position moves only. Status, alarm and tactile data are NOT decoded. Plain C, header-only, no dependencies.
 *
 * RPDO (master -> hand) = 64 bytes = 32 little-endian 16-bit words.
 * TPDO (hand -> master) = 192 bytes.
 * Joints: 1 thumb sweep, 2 thumb flex, 3 little, 4 ring, 5 middle, 6 index (5 and 2 not yet confirmed by eye).
 * Raw position 0..10000 = 0..joint range (deg/range*10000).
 */
#ifndef DH116_PDO_H
#define DH116_PDO_H
#include <stdint.h>
#include <string.h>

#define DH116_RPDO_BYTES 64
#define DH116_TPDO_BYTES 192

/* group k (0..5) in the frame belongs to this joint: thumb flex, thumb sweep, little, ring, middle, index */
static const int DH116_GROUP_JOINT[6] = {2, 1, 6, 5, 4, 3};
static const float DH116_RANGE_DEG[7] = {0, 60, 30, 80, 80, 80, 80};

/* control word: high byte = command, low byte = mode (0x20 = position mode) */
#define DH116_CTRL_ENABLE_PREP 0x0020  /* sent first when enabling   */
#define DH116_CTRL_ENABLED     0x0120
#define DH116_CTRL_DISABLED    0x0220
#define DH116_CTRL_MOVE        0x0820  /* "take the data frame that follows" */
#define DH116_CTRL_HOME_PREP   0x0025
#define DH116_CTRL_HOME_START  0x0425

static inline int dh116_deg_to_raw(int joint, float deg) {
  float r = deg / DH116_RANGE_DEG[joint] * 10000.0f;
  if (r < 0) r = 0;
  if (r > 10000) r = 10000;
  return (int)(r + 0.5f);
}
static inline float dh116_raw_to_deg(int joint, int raw) { return raw / 10000.0f * DH116_RANGE_DEG[joint]; }

static inline void dh116_put16(uint8_t* b, int word, uint16_t v) { b[2 * word] = v & 0xFF; b[2 * word + 1] = v >> 8; }
static inline uint16_t dh116_get16(const uint8_t* b, int off) { return (uint16_t)(b[off] | (b[off + 1] << 8)); }

/* very first frame after init */
static inline void dh116_frame_init(uint8_t out[DH116_RPDO_BYTES]) {
  memset(out, 0, DH116_RPDO_BYTES);
  dh116_put16(out, 0, 0x0200);
  dh116_put16(out, 1, 0x4000);
}
/* control frame: ctrl[1..6] = control word per joint (index 0 unused). Words 7..31 stay 0. */
static inline void dh116_frame_ctrl(uint8_t out[DH116_RPDO_BYTES], const uint16_t ctrl[7]) {
  memset(out, 0, DH116_RPDO_BYTES);
  dh116_put16(out, 0, 0x0601);
  for (int k = 0; k < 6; ++k) dh116_put16(out, 1 + k, ctrl[DH116_GROUP_JOINT[k]]);
}
/* data frame: pos 0..10000, vel (library units, default 20000), cur (default 400); arrays indexed 1..6.
 * velocity word = vel/10 as seen from the library. */
static inline void dh116_frame_data(uint8_t out[DH116_RPDO_BYTES], const int pos[7], const int vel[7], const int cur[7]) {
  memset(out, 0, DH116_RPDO_BYTES);
  dh116_put16(out, 0, 0x060B);
  for (int k = 0; k < 6; ++k) {
    int j = DH116_GROUP_JOINT[k];
    dh116_put16(out, 1 + 3 * k, (uint16_t)pos[j]);
    dh116_put16(out, 2 + 3 * k, (uint16_t)(vel[j] / 10));
    dh116_put16(out, 3 + 3 * k, (uint16_t)cur[j]);
  }
}

/* TPDO joint feedback (frame type byte 0 in 0..4). Returns number of joints decoded, 0 if not a joint frame. */
typedef struct { int pos[7]; int vel[7]; int cur[7]; } dh116_feedback;
static inline int dh116_decode_joints(const uint8_t t[DH116_TPDO_BYTES], dh116_feedback* fb) {
  if (t[0] > 4) return 0;
  int n = t[1]; if (n > 6) n = 6;
  for (int i = 0; i < n; ++i) {
    int j = DH116_GROUP_JOINT[i];
    fb->pos[j] = (int16_t)dh116_get16(t, 2 + 6 * i);
    fb->vel[j] = (int16_t)dh116_get16(t, 4 + 6 * i) * 10;
    fb->cur[j] = (int16_t)dh116_get16(t, 6 + 6 * i);
  }
  return n;
}

/* TPDO tactile (byte 0 == 0x40, byte 1 = number of blocks, usually 6).
 * Blocks of 30 bytes at 2, 32, 62, 92, 122 = thumb, index, middle, ring, little tip;
 * block 5 (palm) starts at 152. Pressure bytes at block start (thumb 5, other fingers 9, palm 22):
 * finger pressure = byte/255. Only checked against the library, not a real hand. */
static inline int dh116_tactile_offset(int sensor_block) { return sensor_block < 5 ? 2 + 30 * sensor_block : 152; }
static inline float dh116_tactile_pressure(const uint8_t t[DH116_TPDO_BYTES], int sensor_block, int idx) {
  return t[dh116_tactile_offset(sensor_block) + idx] / 255.0f;
}
#endif
