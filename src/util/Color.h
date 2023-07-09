#pragma once

#include <cassert>

template<typename T>
struct Color
{
	T r;
	T g;
	T b;
	T a;

	const T& operator[](size_t index) const
	{
		switch (index)
		{
		case 0:	return r;
		case 1:	return g;
		case 2:	return b;
		case 3:	return a;
		}

		// index is bogus
		assert(false);
		return -1;
	}
};

typedef Color<float> Colorf;
typedef Color<uint8_t> Color32;