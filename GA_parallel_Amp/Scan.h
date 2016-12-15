
#pragma once

#include <amp.h>

#define TILE_SIZE                  (256)
#define DEFAULT_NUM_OF_ELEMENTS    (1024)
#define NUM_OF_ELEMENTS            (512)

using namespace concurrency;

template <int _tile_size, typename _type>
void prefix_scan(array_view<_type> a);
template <int _tile_size, typename _type>
void compute_tilewise_scan(array_view<const _type> input, array_view<_type> tilewise_scans, array_view<_type> sums_of_tiles);
template <int _tile_size, typename _type>
void scan_tiled(array_view<const _type> input, array_view<_type> output);

template<typename _type, int _tile_size>
class scan
{
public:
	scan(int size = DEFAULT_NUM_OF_ELEMENTS);
	~scan();
	void execute();
	bool verify();

private:
	std::vector<_type> values;
	std::vector<_type> result;
};