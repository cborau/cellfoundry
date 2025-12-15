// Handy device functions for 3D vector operations. Copy-paste into other function files as needed.
FLAMEGPU_DEVICE_FUNCTION void vec3CrossProd(float &x, float &y, float &z, float x1, float y1, float z1, float x2, float y2, float z2) {
  x = (y1 * z2 - z1 * y2);
  y = (z1 * x2 - x1 * z2);
  z = (x1 * y2 - y1 * x2);
}
FLAMEGPU_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
  x /= divisor;
  y /= divisor;
  z /= divisor;
}
FLAMEGPU_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
  return sqrtf(x * x + y * y + z * z);
}
FLAMEGPU_DEVICE_FUNCTION void vec3Normalize(float &x, float &y, float &z) {
  float length = vec3Length(x, y, z);
  vec3Div(x, y, z, length);
}
FLAMEGPU_DEVICE_FUNCTION float getAngleBetweenVec(const float x1, const float y1, const float z1, const float x2, const float y2, const float z2) {
  float dot_dir = x1 * x2 + y1 * y2 + z1 * z2;
  float cross_x_dir = 0.0;
  float cross_y_dir = 0.0;
  float cross_z_dir = 0.0;
  float angle = 0.0;
  float EPSILON = 0.0000000001;
  vec3CrossProd(cross_x_dir, cross_y_dir, cross_z_dir, x1, y1, z1, x2, y2, z2);
  float det_dir = vec3Length(cross_x_dir, cross_y_dir, cross_z_dir);
  if (fabsf(dot_dir) > EPSILON) {
    angle = atan2f(det_dir, dot_dir);
  }
  else {
    angle = 0.0;
  }
  
  return angle; //in radians
}