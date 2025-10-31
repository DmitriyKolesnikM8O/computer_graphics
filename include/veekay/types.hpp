#pragma once

#include <cstdint>
#include <cmath>

namespace veekay {

union vec2 {
	struct {
		float x;
		float y;
	};

	float elements[2];

	vec2& operator+=(const vec2& other) {
		x += other.x;
		y += other.y;
		return *this;
	}

	vec2& operator+=(float scalar) {
		x += scalar;
		y += scalar;
		return *this;
	}

	vec2& operator-=(const vec2& other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}

	vec2& operator-=(float scalar) {
		x -= scalar;
		y -= scalar;
		return *this;
	}

	vec2& operator*=(const vec2& other) {
		x *= other.x;
		y *= other.y;
		return *this;
	}

	vec2& operator*=(float scalar) {
		x *= scalar;
		y *= scalar;
		return *this;
	}

	vec2& operator/=(const vec2& other) {
		x /= other.x;
		y /= other.y;
		return *this;
	}

	vec2& operator/=(float scalar) {
		x /= scalar;
		y /= scalar;
		return *this;
	}

	vec2 operator+(const vec2& other) const {
		vec2 result = *this;
		return result += other;
	}

	vec2 operator+(float scalar) const {
		vec2 result = *this;
		return result += scalar;
	}

	vec2 operator-(const vec2& other) const {
		vec2 result = *this;
		return result -= other;
	}

	vec2 operator-(float scalar) const {
		vec2 result = *this;
		return result -= scalar;
	}

	vec2 operator-() const {
		return {-x, -y};
	}

	vec2 operator*(const vec2& other) const {
		vec2 result = *this;
		return result *= other;
	}

	vec2 operator*(float scalar) const {
		vec2 result = *this;
		return result *= scalar;
	}

	vec2 operator/(const vec2& other) const {
		vec2 result = *this;
		return result /= other;
	}

	vec2 operator/(float scalar) const {
		vec2 result = *this;
		return result /= scalar;
	}

	float& operator[](size_t index) { return elements[index]; }
	const float& operator[](size_t index) const { return elements[index]; }
};

union vec3 {
	struct {
		float x;
		float y;
		float z;
	};

	float elements[3];

	vec3& operator+=(const vec3& other) {
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	vec3& operator+=(float scalar) {
		x += scalar;
		y += scalar;
		z += scalar;
		return *this;
	}

	vec3& operator-=(const vec3& other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		return *this;
	}

	vec3& operator-=(float scalar) {
		x -= scalar;
		y -= scalar;
		z -= scalar;
		return *this;
	}

	vec3& operator*=(const vec3& other) {
		x *= other.x;
		y *= other.y;
		z *= other.z;
		return *this;
	}

	vec3& operator*=(float scalar) {
		x *= scalar;
		y *= scalar;
		z *= scalar;
		return *this;
	}

	vec3& operator/=(const vec3& other) {
		x /= other.x;
		y /= other.y;
		z /= other.z;
		return *this;
	}

	vec3& operator/=(float scalar) {
		x /= scalar;
		y /= scalar;
		z /= scalar;
		return *this;
	}

	vec3 operator+(const vec3& other) const {
		vec3 result = *this;
		return result += other;
	}

	vec3 operator+(float scalar) const {
		vec3 result = *this;
		return result += scalar;
	}

	vec3 operator-(const vec3& other) const {
		vec3 result = *this;
		return result -= other;
	}

	vec3 operator-(float scalar) const {
		vec3 result = *this;
		return result -= scalar;
	}

	vec3 operator-() const { return {-x, -y, -z}; }

	vec3 operator*(const vec3& other) const {
		vec3 result = *this;
		return result *= other;
	}

	vec3 operator*(float scalar) const {
		vec3 result = *this;
		return result *= scalar;
	}

	vec3 operator/(const vec3& other) const {
		vec3 result = *this;
		return result /= other;
	}

	vec3 operator/(float scalar) const {
		vec3 result = *this;
		return result /= scalar;
	}

	static float dot(const vec3& lhs, const vec3& rhs) {
		return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
	}

	static float squaredLength(const vec3& vector) {
		return dot(vector, vector);
	}

	static float length(const vec3& vector) {
		return std::sqrt(squaredLength(vector));
	}

	static vec3 normalized(const vec3& vector) {
		vec3 result = vector;

		float len = length(vector);
		
		
		if (len < 1e-6) {
			return {0.0f, 0.0f, 0.0f}; 
		}

		result.x /= len;
		result.y /= len;
		result.z /= len;

		return result;
	}

	static vec3 cross(const vec3& lhs, const vec3& rhs) {
		return {
			(lhs.y * rhs.z) - (lhs.z * rhs.y),
			(lhs.z * rhs.x) - (lhs.x * rhs.z),
			(lhs.x * rhs.y) - (lhs.y * rhs.x)
		};
	}

	float& operator[](size_t index) { return elements[index]; }
	const float& operator[](size_t index) const { return elements[index]; }
};

union vec4 {
	struct {
		float x;
		float y;
		float z;
		float w;
	};

	float elements[4];

	vec4& operator+=(const vec4& other) {
		x += other.x;
		y += other.y;
		z += other.z;
		w += other.w;
		return *this;
	}

	vec4& operator-=(const vec4& other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		w -= other.w;
		return *this;
	}

	vec4& operator*=(const vec4& other) {
		x *= other.x;
		y *= other.y;
		z *= other.z;
		w *= other.w;
		return *this;
	}

	vec4& operator/=(const vec4& other) {
		x /= other.x;
		y /= other.y;
		z /= other.z;
		w /= other.w;
		return *this;
	}

	vec4 operator+(const vec4& other) const {
		vec4 result = *this;
		return result += other;
	}

	vec4 operator-(const vec4& other) const {
		vec4 result = *this;
		return result -= other;
	}

	vec4 operator*(const vec4& other) const {
		vec4 result = *this;
		return result *= other;
	}

	vec4 operator/(const vec4& other) const {
		vec4 result = *this;
		return result /= other;
	}

	float& operator[](size_t index) { return elements[index]; }
	const float& operator[](size_t index) const { return elements[index]; }
};

union mat4 {
	float elements[4][4];
	vec4 columns[4];

	static mat4 identity() {
		mat4 result{};

		result[0][0] = 1.0f;
		result[1][1] = 1.0f;
		result[2][2] = 1.0f;
		result[3][3] = 1.0f;
		
		return result;
	}

	static mat4 translation(vec3 vector) {
		mat4 result = mat4::identity();

		result[3][0] = vector.x;
		result[3][1] = vector.y;
		result[3][2] = vector.z;

		return result;
	}

	static mat4 scaling(vec3 vector) {
		mat4 result{};

		result[0][0] = vector.x;
		result[1][1] = vector.y;
		result[2][2] = vector.z;
		result[3][3] = 1.0f;

		return result;
	}

	static mat4 rotation(vec3 axis, float angle) {
		mat4 result{};

		float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

		axis.x /= length;
		axis.y /= length;
		axis.z /= length;

		float sina = sinf(angle);
		float cosa = cosf(angle);
		float cosv = 1.0f - cosa;

		result[0][0] = (axis.x * axis.x * cosv) + cosa;
		result[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
		result[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

		result[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
		result[1][1] = (axis.y * axis.y * cosv) + cosa;
		result[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

		result[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
		result[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
		result[2][2] = (axis.z * axis.z * cosv) + cosa;

		result[3][3] = 1.0f;

		return result;
	}

	static mat4 projection(float fov, float aspect_ratio, float near, float far) {
		mat4 result{};

		const float radians = fov * M_PI / 180.0f;
		const float cot = 1.0f / tanf(radians / 2.0f);

		result[0][0] = cot / aspect_ratio;
		result[1][1] = cot;
		result[2][3] = 1.0f;

		result[2][2] = far / (far - near);
		result[3][2] = (-near * far) / (far - near);

		return result;
	}





	static mat4 look_at(vec3 position, vec3 target, vec3 up) {
		// 1. Z (Forward - от позиции к цели)
		// <--- ИЗМЕНЕНИЕ: target - position
		vec3 z_axis = veekay::vec3::normalized(target - position); 

		// 2. X (Right)
		vec3 x_axis = veekay::vec3::normalized(veekay::vec3::cross(up, z_axis));

		// 3. Y (Up)
		vec3 y_axis = veekay::vec3::cross(z_axis, x_axis); // Y = Z x X

		mat4 result{}; 

		// Строка 0 (X-axis)
		result[0][0] = x_axis.x;
		result[0][1] = x_axis.y;
		result[0][2] = x_axis.z;

		// Строка 1 (Y-axis)
		result[1][0] = y_axis.x;
		result[1][1] = y_axis.y;
		result[1][2] = y_axis.z;

		// Строка 2 (Z-axis). ИСПОЛЬЗУЕМ ИНВЕРТИРОВАННУЮ Z_FORWARD (т.е. Z_BACKWARD)
		// Это стандарт для View Matrix в Vulkan.
		result[2][0] = -z_axis.x; // <--- ИНВЕРСИЯ
		result[2][1] = -z_axis.y; // <--- ИНВЕРСИЯ
		result[2][2] = -z_axis.z; // <--- ИНВЕРСИЯ
		
		// Строка 3 (Трансляция). Используем Z_FORWARD
		result[3][0] = -veekay::vec3::dot(x_axis, position);
		result[3][1] = -veekay::vec3::dot(y_axis, position);
		result[3][2] = -veekay::vec3::dot(z_axis, position); // <--- ВАЖНО: dot с Z_FORWARD
		result[3][3] = 1.0f; 

		return result;
	}





	static mat4 transpose(const mat4& matrix) {
		mat4 result{};

		for (int j = 0; j < 4; ++j) {
			for (int i = 0; i < 4; ++i) {
				result[j][i] = matrix[i][j];
			}
		}

		return result;
	}

	mat4 operator*(const mat4& other) const {
		mat4 result{};

		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 4; i++) {
				for (int k = 0; k < 4; k++) {
					result[j][i] += elements[j][k] * other[k][i];
				}
			}
		}

		return result;
	}

	vec4& operator[](size_t index) { return columns[index]; }
	const vec4& operator[](size_t index) const { return columns[index]; }
};

} // namespace veekay
