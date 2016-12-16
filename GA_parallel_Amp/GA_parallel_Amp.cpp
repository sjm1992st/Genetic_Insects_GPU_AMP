// GA_parallel_Amp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <vector>
#include <amp.h> 


#include <numeric>

using namespace concurrency;

using std::vector;
using std::cout;
using std::endl;

#include "AMP_RNG/inc/amp_tinymt_rng.h"




void randomize_parallel(int seed, vector<float>& source)
{
//	std::cout << "TinyMT   Usage 1: state initialized OUTSIDE kernel (seed=" << seed << ")" << std::endl;

	const int rank = 2;
	extent<rank> e_size(source.size(), 1);
	tinymt_collection<rank> myrand(e_size, seed);

	extent<1> e((int)100);
	array<float, rank> rand_out_data(e_size);


	parallel_for_each(e_size, [=, &rand_out_data](index<2> idx) restrict(amp)
	{
		auto t = myrand[idx];

		// calling below function in loop will give more numbers
		rand_out_data[idx] = t.next_single();
	});


	//std::vector<float> ref_data(e_size.size());
	copy(rand_out_data, source.begin());

/*	for (unsigned i = 0; i < source.size(); i++)
	{
		//	source[i] = rand_out_data.data[i];
		cout << source[i] << " ";
	}

	cout << endl;*/
}


#include "Scan.h"

// Compute prefix of prefix
template <int _tile_size, typename _type>
void prefix_scan(array_view<_type> a)
{
	array<_type> atemp(a.extent);
	scan_tiled<_tile_size>(array_view<const _type>(a), array_view<_type>(atemp));
	copy(atemp, a);
}

template <int _tile_size, typename _type>
void scan_tiled(array_view<const _type> input, array_view<_type> output)
{
	int sz = input.extent[0];
	int number_of_tiles = (sz + _tile_size - 1) / _tile_size;

	// Compute tile-wise scans and reductions
	array<_type> scan_of_sums_of_tiles(number_of_tiles);
	compute_tilewise_scan<_tile_size>(array_view<const _type>(input), array_view<_type>(output), array_view<_type>(scan_of_sums_of_tiles));

	// recurse if necessary
	if (number_of_tiles >  1)
	{
		prefix_scan<_tile_size>(array_view<_type>(scan_of_sums_of_tiles));

		if (sz > 0)
		{
			parallel_for_each(extent<1>(sz), [=, &scan_of_sums_of_tiles](concurrency::index<1> idx) restrict(amp)
			{
				int my_tile = idx[0] / _tile_size;
				if (my_tile == 0)
					output[idx] = output[idx];
				else
					output[idx] = scan_of_sums_of_tiles[my_tile - 1] + output[idx];
			});
		}
	}
}

// Calculate prefix sum for a tile
template <int _tile_size, typename _type>
void compute_tilewise_scan(array_view<const _type> input, array_view<_type> tilewise_scans, array_view<_type> sums_of_tiles)
{
	int sz = input.extent[0];
	int number_of_tiles = (sz + _tile_size - 1) / _tile_size;
	int number_of_threads = number_of_tiles * _tile_size;

	parallel_for_each(extent<1>(number_of_threads).tile<_tile_size>(), [=](tiled_index<_tile_size> tidx) restrict(amp)
	{
		const int tid = tidx.local[0];
		const int globid = tidx.global[0];

		tile_static _type tile[2][_tile_size];
		int in = 0;
		int out = 1;
		if (globid < sz)
			tile[out][tid] = input[globid];
		tidx.barrier.wait();

		for (int offset = 1; offset<_tile_size; offset *= 2)
		{
			// For each iteration, the Y dimension index
			// specifies which index acts as input or output.
			// For each iteration these elements toggle
			in = 1 - in;
			out = 1 - out;

			if (globid < sz)
			{
				if (tid >= offset)
					tile[out][tid] = tile[in][tid - offset] + tile[in][tid];
				else
					tile[out][tid] = tile[in][tid];
			}
			tidx.barrier.wait();
		}
		if (globid < sz)
			tilewise_scans[globid] = tile[out][tid];
		// update prefix sum of the tile to another array_view
		if (tid == _tile_size - 1)
			sums_of_tiles[tidx.tile[0]] = tile[out][tid];
	});
}

template<typename _type, int _tile_size>
scan<_type, _tile_size>::scan(int size)
{
	assert(size%_tile_size == 0);

	for (unsigned i = 0; i < size; i++)
	{
		values.push_back((_type)((rand()) % 127));
		result.push_back(0);
	}
}

template<typename _type, int _tile_size>
scan<_type, _tile_size>::~scan()
{
}

template<typename _type, int _tile_size>
void scan<_type, _tile_size>::execute()
{
	array_view<const _type, 1> a_values(values.size(), values);
	array_view<_type, 1> a_result(result.size(), result);
	scan_tiled<_tile_size>(a_values, a_result);
	copy(a_result, result.begin());
}

template<typename _type, int _tile_size>
bool scan<_type, _tile_size>::verify()
{
	_type sum = (_type)0;

	for (unsigned i = 0; i < values.size(); i++)
	{
		sum += values[i];
		if (result[i] != sum)
		{
			std::cout << i << ": " << sum << " <> " << result[i] << std::endl;
			std::cout << "***SCAN VERIFICATION FAILURE***" << std::endl;
			return false;
		}
	}

	return true;
}
//============================================================================
//
//============================================================================
const int size = 5;

void CppAmpMethod() {
	int aCPP[] = { 1, 2, 3, 4, 5 };
	int bCPP[] = { 6, 7, 8, 9, 10 };
	int sumCPP[size];

	// Create C++ AMP objects.  
	array_view<const int, 1> a(size, aCPP);
	array_view<const int, 1> b(size, bCPP);
	array_view<int, 1> sum(size, sumCPP);
	sum.discard_data();

	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.  
		sum.extent,
		// Define the code to run on each thread on the accelerator.  
		[=](index<1> idx) restrict(amp)
	{

		sum[idx] = a[idx] + b[idx];
	}
	);

	// Print the results. The expected output is "7, 9, 11, 13, 15".  
	for (int i = 0; i < size; i++) {
		std::cout << "sum: " << sum[i] << endl;
	}
}
template<class T>
void vector_copy_template(vector<T> &src, vector<T> &dst)
{
	dst.resize(src.size());
	extent<1> e((int)src.size());
	array_view<T, 1> a_1(e, src);
	array_view<T, 1> a_2(e, dst);
	parallel_for_each(e, [=](index<1> idx) restrict(amp) {
		a_2[idx] = a_1[idx];
	});
	a_2.synchronize();
}

void vector_copy(vector<float> &src, vector<float> &dst)
{
	dst.resize(src.size());
	extent<1> e((int)src.size());
	array_view<float, 1> a_1(e, src);
	array_view<float, 1> a_2(e, dst);
	parallel_for_each(e, [=](index<1> idx) restrict(amp) {
		a_2[idx] = a_1[idx];
	});
	a_2.synchronize();
}

void array_view_copy(array_view<float, 1> &src, array_view<float, 1> &dst)
{
	int size = src.get_extent().size();
	extent<1> e(size);

	parallel_for_each(e, [=](index<1> idx) restrict(amp) {
		dst[idx] = src[idx];
	});
	dst.synchronize();
}

//============================================================================
//
//============================================================================
float sumArray_NaiveAMP(std::vector<float> items){
	auto size = items.size();
	array_view<float, 1> aV(size, items);

	for (int i = 1; i<size; i = 2 * i){
		parallel_for_each(extent<1>(size / 2), [=](index<1> idx) restrict(amp)
		{
			aV[2 * idx*i] = aV[2 * idx*i] + aV[2 * idx*i + i];
		});
	}

	return aV[0];
}



//============================================================================
//
//============================================================================
float max_val_Array_NaiveAMP(std::vector<float> items){
	auto size = items.size();
	array_view<float, 1> aV(size, items);

	for (int i = 1; i<size; i = 2 * i){
		parallel_for_each(extent<1>(size / 2), [=](index<1> idx) restrict(amp)
		{
			float val1 = aV[2 * idx*i];
			float val2 = aV[2 * idx*i + i];
			if (val1 >= val2)
			{
				aV[2 * idx*i] = val1;
			}
			else
			{
				aV[2 * idx*i] = val2;
			}
		});
	}

	return aV[0];
}

//============================================================================
//
//============================================================================
float min_val_Array_NaiveAMP(std::vector<float> items){
	auto size = items.size();
	array_view<float, 1> aV(size, items);

	for (int i = 1; i<size; i = 2 * i){
		parallel_for_each(extent<1>(size / 2), [=](index<1> idx) restrict(amp)
		{
			float val1 = aV[2 * idx*i];
			float val2 = aV[2 * idx*i + i];
			if (val1 <= val2)
			{
				aV[2 * idx*i] = val1;
			}
			else
			{
				aV[2 * idx*i] = val2;
			}
		});
	}

	return aV[0];
}
/*float sumArray_NaiveAMP_AV(array_view<float, 1> aV){
auto size = aV.get_extent().size();

for (int i = 1; i<size; i = 2 * i){
parallel_for_each(extent<1>(size / 2), [=](index<1> idx) restrict(amp)
{
aV[2 * idx*i] = aV[2 * idx*i] + aV[2 * idx*i + i];
});
}

return aV[0];
}*/

//============================================================================
//
//============================================================================
float mean(vector<float> vA) {

	float size = vA.size();

	float total = 0;
	total = sumArray_NaiveAMP(vA); // this destroys the original array
	float avg = total / size;

	return avg;
}

//============================================================================
//
//============================================================================
/*float mean_AV(array_view<float, 1> aV) {

float size = aV.get_extent().size();
vector<float> vA(int(size));
array_view<float, 1> a2((int)size, vA);

array_view_copy(aV, a2);

float total = 0;
total = sumArray_NaiveAMP_AV(a2); // this destroys the original array
float avg = total / size;

return avg;
}*/



//============================================================================
//
//============================================================================
float variance(vector<float> vA) {

	float size = vA.size();


	extent<1> e((int)size);
	vector<float> vDistance(size);

	array_view<float, 1> a(e, vA);
	array_view<float, 1> distance(e, vDistance);

	float mean_val = mean(vA);

	parallel_for_each(e, [=](index<1> idx) restrict(amp) {
		distance[idx] = (a[idx] - mean_val)*(a[idx] - mean_val);
	});

	distance.synchronize();
	float dispersion = sumArray_NaiveAMP(vDistance);
	return dispersion / size;
}


//============================================================================
//
//============================================================================
float variance(float mean_val, vector<float> vA) {

	float size = vA.size();

	extent<1> e((int)size);
	vector<float> vDistance((int)size);

	array_view<float, 1> a(e, vA);
	array_view<float, 1> distance(e, vDistance);

	parallel_for_each(e, [=](index<1> idx) restrict(amp) {
		distance[idx] = (a[idx] - mean_val)*(a[idx] - mean_val);
	});

	distance.synchronize();
	float dispersion = sumArray_NaiveAMP(vDistance);
	return dispersion / size;
}
//============================================================================
//
//============================================================================
float standard_deviation(float mean_val, vector<float> vA) {

	return sqrt(variance(mean_val, vA));
}

//============================================================================
//
//============================================================================
float standard_deviation(vector<float> vA) {

	return sqrt(variance(vA));
}



//============================================================================
//
//============================================================================
/*float variance(array_view<float, 1> a) {

float size = a.get_extent().size();
extent<1> e((int)size);
vector<float> vDistance(size);
vector<float> a2(size);
array_view<float, 1> a_2(e, a2);

array_view_copy(a, a_2);

float mean_val = mean_AV(a_2);

array_view<float, 1> distance(e, vDistance);

parallel_for_each(e, [=](index<1> idx) restrict(amp) {
distance[idx] = (a[idx] - mean_val)*(a[idx] - mean_val);
});

distance.synchronize();
float dispersion = sumArray_NaiveAMP(vDistance);
return dispersion / size;
}*/

//============================================================================
//
//============================================================================
/*float variance(float mean_val, array_view<float, 1> a) {

float size = a.get_extent().size();
extent<1> e((int)size);
vector<float> vDistance(size);

array_view<float, 1> distance(e, vDistance);

parallel_for_each(e, [=](index<1> idx) restrict(amp) {
distance[idx] = (a[idx] - mean_val)*(a[idx] - mean_val);
});

distance.synchronize();
float dispersion = sumArray_NaiveAMP(vDistance);
return dispersion / size;
}*/

//============================================================================
//
//============================================================================
/*float standard_deviation(array_view<float, 1> a) {

return sqrt(variance(a));
}*/

//============================================================================
//
//============================================================================
/*float standard_deviation(float mean_val, array_view<float, 1> a) {

return sqrt(variance(mean_val,a));
}*/

void standardize(vector<float> &vA) {

	float mean_val = mean(vA);
	float std_dev = standard_deviation(vA);

	extent<1> e(vA.size());
	array_view<float, 1> a(e, vA);

	parallel_for_each(e, [=](index<1> idx) restrict(amp) {
		a[idx] = (a[idx] - mean_val) / std_dev;
	});

	a.synchronize();
}

/*array_view<float, 1>& standardize(array_view<float, 1> &a) {

float mean_val = mean_AV(a);
float std_dev = standard_deviation(mean_val, a);
float size = a.get_extent().size();
extent<1> e = a.get_extent();



parallel_for_each(e, [=](index<1> idx) restrict(amp) {
a[idx] = (a[idx] - mean_val) / std_dev;
});

a.synchronize();

return a;

}
*/

//============================================================================
//
//============================================================================
template<class Scalar>
static inline Scalar RandomScalar(Scalar min, Scalar max)
{
	Scalar r = (Scalar)rand() / (Scalar)RAND_MAX;
	return min + r * (max - min);
}


float RandomFloat(float min, float max)
{
	float r = (float)rand() / (float)RAND_MAX;
	return min + r * (max - min);
}

class Rand
{
public:
	float min_value = -1.0;
	float max_value = 1.0;

	Rand(float min_val, float max_val)
		: min_value(min_val), max_value(max_val) {}

	void operator()(float& value)
	{
		value = RandomFloat(min_value, max_value);
	}
};

//============================================================================
//
//============================================================================
vector<float> std_normal_distribution(int num, float min_val, float max_val) {

	vector<float> vA(num);

	Rand my_rand(min_val, max_val);
	std::for_each(vA.begin(), vA.end(), my_rand);

	return vA;

}

float std_deviation(float min_val, float max_val)
{
	return (max_val - min_val) / 4;
}

#include <algorithm>

//#include "Utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
//#include "Utils.h"

inline float RandomInt(int min, int max)
{
	float r = (float)rand() / (float)RAND_MAX;
	return (int)((float)min + r * float(max - min));
}

using std::vector;





void SetBit(int &dna, int idx)
{
	dna |= (1 << idx);
}

bool CheckBit(int &dna, int idx)
{
	return dna & (1 << idx);
}

void ResetBit(int &dna, int idx)
{
	dna &= ~(1 << idx);
}


namespace SIMPLE_GA {

	class InsectChromosome1
	{
	public:

		int dna = 0;

		float fitness = 0.0f;

		int target = 0;
		int numTraits = 8;
		int numGenes = 25;

		int ANTENNAE_MASK = 0;
		int HEAD_MASK = 0;
		int WINGS_MASK = 0;
		int BODY_MASK = 0;
		int FEET_MASK = 0;
		int BODY_COLOR_MASK = 0;
		int SIZE_MASK = 0;
		int HEAD_COLOR_MASK = 0;

		int antennae_start = 0;
		int head_start = 4;
		int wing_start = 6;
		int body_start = 10;
		int feet_start = 16;
		int body_color_start = 17;
		int size_start = 20;
		int head_color_start = 22;
		int head_color_end = 25;

		InsectChromosome1()
		{
			// compute the masks
			for (int i = antennae_start; i < head_start; i++)		SetBit(ANTENNAE_MASK, i);
			for (int i = head_start; i < wing_start; i++)			SetBit(HEAD_MASK, i);
			for (int i = wing_start; i < body_start; i++)			SetBit(WINGS_MASK, i);
			for (int i = body_start; i < feet_start; i++)			SetBit(BODY_MASK, i);
			for (int i = feet_start; i < body_color_start; i++)		SetBit(FEET_MASK, i);
			for (int i = body_color_start; i < size_start; i++)		SetBit(BODY_COLOR_MASK, i);
			for (int i = size_start; i < head_color_start; i++)		SetBit(SIZE_MASK, i);
			for (int i = head_color_start; i < head_color_end; i++) SetBit(HEAD_COLOR_MASK, i);
		}

		void SetAntennae(int choice)
		{
			dna |= (ANTENNAE_MASK & (choice));
		}
		void SetHead(int choice)
		{
			dna |= (HEAD_MASK & (choice << head_start));
		}
		void SetWing(int choice)
		{
			dna |= (WINGS_MASK & (choice << wing_start));
		}
		void SetBody(int choice)
		{
			dna |= (BODY_MASK & (choice << body_start));
		}
		void SetFeet(int choice)
		{
			dna |= (FEET_MASK & (choice << feet_start));
		}
		void SetBodyColor(int choice)
		{
			dna |= (BODY_COLOR_MASK & (choice << body_color_start));
		}
		void SetSize(int choice)
		{
			dna |= (SIZE_MASK & (choice << size_start));
		}
		void SetHeadColor(int choice)
		{
			dna |= (HEAD_COLOR_MASK & (choice << head_color_start));
		}
		int GetAntennae()
		{
			return dna & (ANTENNAE_MASK);
		}
		int GetHead()
		{
			return ((dna & HEAD_MASK) >> head_start);
		}
		int GetWing()
		{
			return ((dna & WINGS_MASK) >> wing_start);
		}
		int GetBody()
		{
			return ((dna & BODY_MASK) >> body_start);
		}
		int GetFeet()
		{
			return ((dna & FEET_MASK) >> feet_start);
		}
		int GetBodyColor()
		{
			return ((dna & BODY_COLOR_MASK) >> body_color_start);
		}
		int GetSize()
		{
			return ((dna & SIZE_MASK) >> size_start);
		}
		int GetHeadColor()
		{
			return ((dna & HEAD_COLOR_MASK) >> head_color_start);
		}



		void Fitness() {
			int score = 0;
			for (int i = 0; i < this->numGenes; i++) {
				//Is the character correct ?
				if (CheckBit(dna, i))
				{
					if (CheckBit(target, i))
						score++;
				}
				else
				{
					if (!CheckBit(target, i))
						score++;
				}

			}
			//Fitness is the percentage correct.
			fitness = float(score) / (float)numGenes;
			//fitness = fitness * fitness;
		}

		void mutate(float mutationRate) {

			if (RandomFloat(0.0, 1.0) < mutationRate) {
				int gene_to_mutate = RandomInt(0, 25);

				if (dna & (1 << gene_to_mutate))
				{
					dna |= (0 << gene_to_mutate);
				}
				else dna |= (1 << gene_to_mutate);
			}
		}

		int crossoverSinglePoint(int dnaB)
		{
			int cross_point = RandomInt(0, 25);

			int output = 0;

			for (int i = 0; i < cross_point; i++)
			{
				if (CheckBit(dna, i))
					SetBit(output, i);
			}
			for (int i = cross_point; i < 32; i++)
			{
				if (CheckBit(dnaB, i))
					SetBit(output, i);
			}

			return output;
		}

		int crossoverTwoPoint(int dnaB)
		{
			int cross_point_1 = RandomInt(0, 25);
			int cross_point_2 = 0;

			if (cross_point_1 > 12)
			{
				cross_point_2 = RandomInt(0, 12);
				// swap them
				int temp = cross_point_1;
				cross_point_1 = cross_point_2;
				cross_point_2 = temp;
			}
			else
			{
				cross_point_2 = RandomInt(12, 25);
			}

			int output = 0;

			for (int i = 0; i < cross_point_1; i++)
			{
				if (CheckBit(dna, i))
					SetBit(output, i);
			}

			for (int i = cross_point_1; i < cross_point_2; i++)
			{
				if (CheckBit(dnaB, i))
					SetBit(output, i);
			}
			for (int i = cross_point_2; i < 32; i++)
			{
				if (CheckBit(dna, i))
					SetBit(output, i);
			}

			return output;

		}

		int uniformCrossover(int dnaB)
		{
			int mask = 0;
			for (int i = 0; i < 25; i++)
			{
				SetBit(mask, RandomInt(0, 2));
			}

			int output = 0;

			output = (dna & mask) | (dnaB & ~mask);

			return output;
		}

		int heuristicCrossover(int dnaB, float alpha)
		{
			int output = 0;

			output = (int)((float)dna * alpha) + (1 - alpha) * (float)dnaB;

			return output;
		}

		void SetTarget(int t){ target = t; }

	};



	class Population
	{
	public:

		vector<InsectChromosome1> population;

		int population_size = 200;
		float mutationRate = 0.03;
		float sumFitness = 0.0f;
		float best_fitness = 0.0f;
		int best_index = 0;
		bool match = false;
		int target = 0;

		InsectChromosome1 temp_mating_insect_storage;

		std::vector<int> mating_pool;
		std::vector<int> last_phase_mating_pool;

		std::vector<int> *mating_pool_ptrA = 0;
		std::vector<int> *mating_pool_ptrB = 0;
		std::vector<int> *temp_mating_pool_ptr = 0;

		std::vector<float> fitness;
		std::vector<int> index_vector;

		std::ofstream myfile;
		vector<float> CDF;
		vector<float> selection1;
		vector<float> selection2;
		vector<float> rand;

		float * cdf_matrix = 0;
		vector< vector<float> > row_vectors;
		vector< float > row_vec1;
		vector< float > row_vec2;
		vector<int> pop_crossover;
		Population()
		{
			InsectChromosome1 c1;

			// setup the target insect with predetermined choices

			c1.SetHead(2);
			c1.SetAntennae(4);
			c1.SetBody(34);
			c1.SetWing(5);
			c1.SetFeet(1);
			c1.SetBodyColor(3);
			c1.SetSize(2);
			c1.SetHeadColor(1);

			target = c1.dna;
		}

		Population(int size)
		{
			population_size = size;


			// setup the target insect with predetermined choices
			InsectChromosome1 c1;

			c1.SetHead(2);
			c1.SetAntennae(4);
			c1.SetBody(34);
			c1.SetWing(5);
			c1.SetFeet(1);
			c1.SetBodyColor(3);
			c1.SetSize(2);
			c1.SetHeadColor(1);

			target = c1.dna;
		}

		~Population()
		{
			if (population.size() > 0)
			{

				population.clear();

			}

			myfile.close();
		}

		int GetBest()
		{
			return this->last_phase_mating_pool[best_index];
		}

		void Initialize()
		{
			myfile.open("example_stats6.csv");

			cdf_matrix = new float[population_size*population_size];


			for (int i = 0; i < population_size; i++) {

				//Initializing each member of the population
				InsectChromosome1 chromo;// = new InsectChromosome1();

				chromo.SetTarget(target);
				population.push_back(chromo);
				mating_pool.push_back(chromo.dna);
				last_phase_mating_pool.push_back(chromo.dna);
				fitness.push_back(0.0);
				index_vector.push_back(i);
				CDF.push_back(0.0);
				rand.push_back(0.0);

				selection1.push_back(0.0);
				selection2.push_back(0.0);
				myfile << i << ",";

				row_vec1.push_back(0.0);
				row_vec2.push_back(0.0);


				pop_crossover.push_back(0);
				vector<float> v;
				for (int i = 0; i < population_size; i++)
				{
					v.push_back(0.0);
				}
				row_vectors.push_back(v);
			}
			myfile << "mean," << std::endl;

			mating_pool_ptrA = &mating_pool;
			mating_pool_ptrB = &last_phase_mating_pool;
		}
		//============================================================================
		//
		//============================================================================
		void max_fitness_val_AMP(){

			// copy the vector
			vector<int> backup_indexes;
			vector_copy_template<int>(this->index_vector, backup_indexes);

			auto size = population.size();
			array_view<float, 1> aV(size, fitness);
			array_view<int, 1> index_vector_aV(size, backup_indexes);

			for (int i = 1; i<size; i = 2 * i){
				parallel_for_each(extent<1>(size / 2), [=](index<1> idx) restrict(amp)
				{
					float val1 = aV[2 * idx*i];
					float val2 = aV[2 * idx*i + i];

					int  index1 = index_vector_aV[2 * idx*i];
					int index2 = index_vector_aV[2 * idx*i + i];

					if (val1 >= val2)
					{
						index_vector_aV[2 * idx*i] = index1;
					}
					else
					{
						index_vector_aV[2 * idx*i] = index2;
					}
				});
			}
		
			this->best_index = index_vector_aV[0];
			this->best_fitness = fitness[best_index];
		}


		

		//https://blogs.msdn.microsoft.com/nativeconcurrency/2012/03/13/scan-using-c-amp/
		void compute_CDF()
		{
			//auto size = population.size();
			extent<1> size((int)population.size());
			array_view<float, 1> f_aV(size, fitness);
			array_view<float, 1> cdf_aV(size, this->CDF);

			float sum_fitness_local = this->sumFitness;

			parallel_for_each(size, [=](index<1> idx) restrict(amp) {
				cdf_aV[idx] = f_aV[idx];
			});

			cdf_aV.synchronize();


			cout << endl;

			prefix_scan<10, float>(cdf_aV);

			int local_population_size = population_size;
			parallel_for_each(size, [=](index<1> idx) restrict(amp) {
				
				cdf_aV[idx] = cdf_aV[idx] / cdf_aV[local_population_size - 1];
			});
			cdf_aV.synchronize();
			/*for (int i = 0; i < CDF.size(); i++)
			{
			cout << CDF[i] << " ";
			}
			cout << endl;
			*/

			


		}

		void SelectParent(vector<float> &row_vec)
		{
			{
				


				array_view<float, 2> a1(population.size(), population.size(), cdf_matrix);

				array_view<float, 2> a2(CDF.size(), 1, CDF);



				array_view<float, 2> rand_n(CDF.size(), 1, rand);

				parallel_for_each(
					a1.extent, [=](index<2> idx) restrict(amp)
				{
					int row = idx[0];
					int col = idx[1];

					float row_indexes_col = rand_n(row, 0);// (float)row / 50;

					float row_val = a2(row, 0);

					if (row == 0)
					{
						a1(row, col) = 0.0;
					}
					else
					{
						if (col > 0)
						{
							float a2_col = a2(col, 0);
							float a2_col_sub1 = a2(col - 1, 0);
							if ((a2_col >= row_indexes_col) && (a2_col_sub1 < row_indexes_col))
							{
								a1(row, col) = col;
							}
							else
							{
								a1(row, col) = 0.0;
							}
						}
						else
						{
							a1(row, col) = 0.0;

						}
					}

					//			a1(row, col) = row * 2 + col;
				});
				a1.synchronize();



				for (int j = 0; j < CDF.size(); j++)
				{
					array_view<float, 2> row_a2(CDF.size(), 1, row_vectors[j]);
					//CopyRowToColumn(a1, j, row_a2, 50);

					parallel_for_each(
						a1.extent, [=](index<2> idx) restrict(amp)
					{
						int row = idx[0];
						int col = idx[1];


						if (row == j)
						{
							float old_val = a1(j, col);
							row_a2(col, 1) = old_val;
						}
					});

				}

				for (int j = 0; j < CDF.size(); j++)
				{
					const unsigned window_width = 10;

					unsigned element_count = static_cast<unsigned>(CDF.size());
					assert(element_count != 0); // Cannot reduce an empty sequence.

					// Using array as temporary memory.
					array<float, 1> a_1(element_count, row_vectors[j].begin());

					// Takes care of the sum of tail elements.
					float tail_sum = 0.f;
					if ((element_count % window_width) != 0 && element_count > window_width)
					{
						tail_sum = std::accumulate(row_vectors[5].begin() + ((element_count - 1) / window_width) * window_width, row_vectors[5].end(), 0.f);
					}
					array_view<float, 1> av_tail_sum(1, &tail_sum);

					// Each thread reduces window_width elements.
					unsigned prev_s = element_count;
					for (unsigned s = element_count / window_width; s > 0; s /= window_width)
					{
						parallel_for_each(extent<1>(s), [=, &a_1](index<1> idx) restrict(amp)
						{
							float sum = 0.f;
							for (unsigned i = 0; i < window_width; i++)
							{
								sum += a_1[idx + i * s];
							}
							a_1[idx] = sum;

							// Reduce the tail in cases where the number of elements is not divisible.
							// Note: execution of this section may negatively affect the performance.
							// In production code the problem size passed to the reduction should
							// be a power of the window_width. Please refer to the blog post for more
							// information.
							if ((idx[0] == s - 1) && ((s % window_width) != 0) && (s > window_width))
							{
								for (unsigned i = ((s - 1) / window_width) * window_width; i < s; i++)
								{
									av_tail_sum[0] += a_1[i];
								}
							}
						});
						prev_s = s;
					}

					// Perform any remaining reduction on the CPU.
					std::vector<float> result(prev_s);
					copy(a_1.section(0, prev_s), result.begin());
					av_tail_sum.synchronize();

					row_vec[j] = std::accumulate(result.begin(), result.end(), tail_sum);

				//	cout << "RESULT WAS: " << row_vec[j] << endl;
				//	cout << endl;

				}


			//	for (int j = 0; j < CDF.size(); j++)
			//		cout << row_vec[j] << " ";

			//	cout << endl;

			}
		}

		

		void ChooseMates()
		{
			compute_CDF();
			randomize_parallel(RandomInt(0, 65535), rand);

			this->SelectParent(row_vec1);

			randomize_parallel(RandomInt(0, 65535), rand);
			this->SelectParent(row_vec2);


			extent<1> e((int)population_size);
			//vector<int> mating(size);

			array_view<int, 1> a(e, last_phase_mating_pool);
			array_view<int, 1> a1(e, mating_pool);
			array_view<float, 1> row_vec_1(e, row_vec1);
			array_view<float, 1> row_vec_2(e, row_vec2);

		

			
			for (int i = 0; i < population_size; i++)
				pop_crossover[i] = RandomInt(0, 25);

			array_view<int, 1> cross_point(e, pop_crossover);

			parallel_for_each(e, [=](index<1> idx) restrict(amp) {

				// forced to unwrap the Fitness function, 
				// and as each bit is checked rather than a bool
				// we have to loop here (but this is in parallel for each
				// population member)

				int i_1 = row_vec_1[idx];
				int i_2 = row_vec_2[idx];

				int dnaA = a1[i_1];
				int dnaB = a1[i_2];
				int cp = cross_point[idx];

				int output = 0;

				for (int x = 0; x < cp; x++)
				{
					if (dnaA & (1 << x))
						output |= (1 << x);
				}
				for (int x = cp; x < 32; x++)
				{
					if (dnaB & (1 << x))
						output |= (1 << x);
				}
				a[idx] = output;
			});
			a.synchronize();

			//for (int i = 0; i < population_size; i++)
			//	print_dna(last_phase_mating_pool[i]);
		}


		void Mutate()
		{
			for (int i = 0; i < population_size; i++)
				pop_crossover[i] = RandomInt(0, 25);

			randomize_parallel(RandomInt(0, 65535), rand);

			extent<1> e((int)population_size);
			array_view<int, 1> mutate_point(e, pop_crossover);

			array_view<int, 1> a(e, last_phase_mating_pool);
			array_view<float, 1> rand_n(e, rand);

			float mutation_rate_local = this->mutationRate;

			parallel_for_each(e, [=](index<1> idx) restrict(amp) {

				if (rand_n[idx] < mutation_rate_local)
				{
					int gene_to_mutate = mutate_point[idx];

					int dna = a[idx];

					if (dna & (1 << gene_to_mutate))
					{
						a[idx] |= (0 << gene_to_mutate);
					}
					else a[idx] |= (1 << gene_to_mutate);
				}
			});

			a.synchronize();

	
		}

		//
		// Worst case. this is O(n^2) because we have the inner loop in the second phase
		//
		void Update()
		{
			int best = 0;
			best_index = 0;
			best_fitness = 0.0f;
			sumFitness = 0.0f;

			float sum_last_fitness = sumFitness;

			float population_size = population.size();


			extent<1> e((int)population_size);
			//vector<int> mating(size);

			array_view<int, 1> a(e, last_phase_mating_pool);
			array_view<int, 1> mating(e, mating_pool);
			array_view<float, 1> fit_av(e, fitness);
			//float mean_val = mean(vA);

			int targ = this->target;

			//evaluate fitness
			parallel_for_each(e, [=](index<1> idx) restrict(amp) {

				// forced to unwrap the Fitness function, 
				// and as each bit is checked rather than a bool
				// we have to loop here (but this is in parallel for each
				// population member)
				int score = 0;
				for (int i = 0; i < 25; i++) {
					//Is the character correct ?
					
					if (a[idx] & (1 << i))
					{
						if (targ & (1 << i))
							score++;
					}
					else
					{
						if (!(targ & (1 << i)))
							score++;
					}

				}
				//a[idx].fitness = 
				fit_av[idx] = float(score) / 25.0;
				//sumFitness += a[idx].fitness;
				mating[idx] = a[idx];

			});

			// the array views must be synchronized
			fit_av.synchronize();
			mating.synchronize();

			max_fitness_val_AMP();

			this->sumFitness = sumArray_NaiveAMP(fitness);

			//fit_av.synchronize();
		
		//	compute_CDF();

			int best_value = this->GetBest();

			float mean_fitness = sumFitness / population_size;
			
			static int epoch = 0;
			//myfile << "EPOCH " << epoch << ", ";
			//	myfile << "SUM FITNESS: " << sumFitness << ", SUM FITNESS*100: " << sumFitness*100 << std::endl;


/*			for (int i = 0; i < population.size(); i++) {

				//std::string s = string_dna(population[i]->dna);
				myfile << population[i].fitness << ",";// << std::endl;

				if (i == this->best_index)
				{

					continue;
				}

				std::int32_t a = this->ChooseParent(-1);
				std::int32_t b = this->ChooseParent(a);

				if (a == b) if (a <= population.size() - 2)b = a + 1; else b = 0;

				int32_t partnerA = mating_pool[a];
				int32_t partnerB = mating_pool[b];

				temp_mating_insect_storage.dna = partnerA;

				//Step 3a: Crossover  (Note some of the other crossover operations have not been implemented)
				population[i].dna = temp_mating_insect_storage.crossoverSinglePoint(partnerB);

				last_phase_mating_pool[i] = population[i].dna;
			}*/
			this->ChooseMates();
			this->Mutate();

			myfile << mean_fitness << "," << std::endl;

			epoch++;
			//mating_pool.clear();
		}




		// 
		int ChooseParent(int parent_to_skip)
		{
			int randSelector = (int)RandomFloat(0, sumFitness) * 100;

			int memberID = 0;
			int partialSum = 0;

			for (int i = 0; i < population.size(); i++) {

				int n = (int)(fitness[i] * 100);
				if (partialSum < randSelector && partialSum + n >= randSelector)
				{
					if (i == parent_to_skip)
					{
						if (i + 1 == population.size()) return best_index; // more breeding with population best
						else return i + 1;
					}
					else
					{
						return i;
					}
				}
				partialSum += n;
			}

			return this->best_index;
		}


		void Delete()
		{


			population.clear();
		}

		void print_dna(int g)
		{
			for (int i = 0; i < 32; i++)
			{
				if (CheckBit(g, i)) std::cout << "1";
				else std::cout << "0";
			}
		}

		std::string string_dna(int g)
		{
			std::string ret = "";
			for (int i = 0; i < 32; i++)
			{
				if (CheckBit(g, i)) ret += "1";
				else ret += "0";
			}
			return ret;
		}
	};


};

using namespace SIMPLE_GA;

//
//  Object: SimpleExample
//  abstract base class
//
class SimpleExample
{
protected:
	bool m_bDeleted = false;
public:

	SimpleExample(){}
	~SimpleExample(){
		if (m_bDeleted == false)
			this->Delete();
	}

	virtual void Initialize(){}
	virtual void Update(){};
	virtual void Draw(){};
	virtual void Delete(){};
	virtual void keyboard(unsigned char key, int x, int y){};
	virtual void keyboardup(unsigned char key, int x, int y){};
	virtual void Special(int key, int x, int y){};

	virtual void mouse(int button, int state, int x, int y){}
};

//
//  Object: SimpleExample
//  Simply runs the code 
//
class SimpleGA : public SimpleExample
{
public:

	int population_size;


	float mutationRate = 0.01;

	bool solved = false;


	Population population3;



	void SetPopulationSize(int n)
	{
		this->population_size = n;
	}

	void Initialize()
	{
		// seed random number generator ...
		int randSelector = (int)RandomFloat(0, 23);
		randSelector = (int)RandomFloat(0, 23);
		randSelector = (int)RandomFloat(0, 23);

		population3.Initialize();
	}

	void Draw() {}

	void Update() {

		if (solved == false)
		{
			population3.Update();

			static int generation_count = 0;
			std::cout << "generation" << generation_count++ << "    " << population3.best_fitness << std::endl;//

			population3.print_dna(population3.GetBest());
			std::cout << std::endl;
			population3.print_dna(population3.target);


			std::cout << std::endl;

			if (population3.best_fitness > 0.97f)
			{
				solved = true; // print statistics
				int best = population3.best_index;
				population3.population[best].dna = population3.GetBest();
				std::cout << "****************************************************" << std::endl;
				std::cout << std::endl << "Insect Chosen Characteristics" << std::endl;
				
				
				if (best < population3.population.size())
				{
					std::cout << "Antennae Type: " << population3.population[best].GetAntennae() << std::endl;
					std::cout << "Head Type: " << population3.population[best].GetHead() << std::endl;
					std::cout << "Wing Type: " << population3.population[best].GetWing() << std::endl;
					std::cout << "Body Type: " << population3.population[best].GetBody() << std::endl;
					std::cout << "Size Type: " << population3.population[best].GetSize() << std::endl;
					std::cout << "Body Color: " << population3.population[best].GetBodyColor() << std::endl;
					std::cout << "Feet: " << population3.population[best].GetFeet() << std::endl;
					std::cout << "Head Color: " << population3.population[best].GetHeadColor() << std::endl;
					std::cout << std::endl << "****************************************************" << std::endl;
				}
			}
		}
	}

	void keyboard(unsigned char key, int x, int y){}
	void keyboardup(unsigned char key, int x, int y){}
	void Special(int key, int x, int y){};

	void mouse(int button, int state, int x, int y){}

	void Delete(){

		population3.Delete();
		//mating_pool.clear();
		this->m_bDeleted = true;

	}
};




/*
ANTENNAE_MASK 00001111111111111111111111111111
HEAD_MASK	  11110001111111111111111111111111
WINGS_MASK	  11111110000111111111111111111111
BODY_MASK	  11111111111000000111111111111111
FEET_MASK	  11111111111111111011111111111111
BCOLOR_MASK	  11111111111111111100011111111111
SIZE_MASK	  11111111111111111111100111111111
HCOLOR_MASK	  11111111111111111111111000111111
*/


//----------------------------------------------------------------------------
// This is an implementation of the reduction algorithm using a simple
// parallel_for_each. Multiple kernel launches are required to synchronize
// memory access among threads in separate tiles.
//----------------------------------------------------------------------------
float reduction_simple_1( vector<float>& source)
{
	assert(source.size() <= UINT_MAX);
	unsigned element_count = static_cast<unsigned>(source.size());
	assert(element_count != 0); // Cannot reduce an empty sequence.
	if (element_count == 1)
	{
		return source[0];
	}

	// Using array, as we mostly need just temporary memory to store
	// the algorithm state between iterations and in the end we have to copy
	// back only the first element.
	//array<float, 1> a(element_count, source.begin());
	extent<1> e((int)source.size());
	// Takes care of odd input elements – we could completely avoid tail sum
	// if we would require source to have even number of elements.
	float tail_sum = (element_count % 2) ? source[element_count - 1] : 0;
	array_view<float, 1> av_tail_sum(1, &tail_sum);
	array_view<float, 1> a(e, source);

	// Each thread reduces two elements.
	for (unsigned s = 0; s < element_count/2; s++)
	{
		parallel_for_each(extent<1>(s), [=](index<1> idx) restrict(amp)
		{
			a[idx] = a[idx] + a[idx + s];

			// Reduce the tail in cases where the number of elements is odd.
			if ((idx[0] == s - 1) && (s & 0x1) && (s != 1))
			{
				av_tail_sum[0] += a[s - 1];
			}
		});
		a.synchronize();
	}
	
	// Copy the results back to CPU.
	std::vector<float> result(1);
	result[0] = a[0];
	//copy(a.section(0, 1), result.begin());
	av_tail_sum.synchronize();
	

	return result[0] + tail_sum;
}


void CopyRowToColumn(array_view<float, 2> a1, int t_row, array_view<float, 2> a2, int size)
{
//	array_view<float, 2> a1(50, 50, index_vec);

//	array_view<float, 2> a2(50, 1, test_vec);



	parallel_for_each(
		a1.extent, [=](index<2> idx) restrict(amp)
	{
		int row = idx[0];
		int col = idx[1];


		if ( row == t_row )
		{ 
		float old_val = a1(t_row, col);
		a2(col, 1) = old_val;
		}
	});
//	a2.synchronize();
}

//============================================================================
//
//============================================================================
int _tmain(int argc, _TCHAR* argv[])
{
	CppAmpMethod();

	vector<float> arr;
	for (int i = 0; i < 250; i++)
		arr.push_back(float(i));

	std::random_shuffle(arr.begin(),arr.end());
	cout << "max value of array: " << max_val_Array_NaiveAMP(arr) << endl;

	vector<float> rand_arr= std_normal_distribution(1000, -3.0, 19.2);

	cout << "mean of array: " << mean(rand_arr) << " standard deviation of array: " << standard_deviation(rand_arr) << endl;

	cout << "std deviation method 2: " << std_deviation(0.0, 1.0) << endl;

	standardize(rand_arr);
	float mean_val = mean(rand_arr);
	cout << "mean of array: " << mean_val << " standard deviation of array: " << standard_deviation(mean_val, rand_arr) << endl;
	cout << "max of array: " << max_val_Array_NaiveAMP(rand_arr) << endl;
	

	SimpleGA *example = new  SimpleGA();
	example->Initialize();
	while (example->solved == false)
		example->Update();
	example->Delete();
	delete example;


	cout << endl; 
	cout << "Reduction test" << endl;
	vector<float> test_vec;
	for (int i = 0; i < 50; i++)
	{
		float sigmoid = 1 / (1 + std::exp(-0.125 * (float)i));
		test_vec.push_back(sigmoid);
		cout << sigmoid << " ";
	}
	cout << endl;


	extent<1> e((int)test_vec.size());
	// Takes care of odd input elements – we could completely avoid tail sum
	// if we would require source to have even number of elements.

	cout << endl;
	cout << endl;
	array_view<float, 1> a(e, test_vec);
	prefix_scan<10, float>(a);

	for (int i = 0; i < 50; i++)
	{
		cout << test_vec[i] << " ";
		test_vec[i] /= test_vec[50 - 1];
	}
	cout << endl;
	cout << endl;
	
	float* index_vec = new float[50*50];
	
	vector< float > random_vec;
	vector< float > row_vec;

	

	for (int i = 0; i < 50; i++)
	{
		cout << test_vec[i] << " ";
		random_vec.push_back(0.0);
		row_vec.push_back(0.0);
		//test_vec[i] /= test_vec[50 - 1];
	}

	{


		cout << endl;
		cout << endl;
		randomize_parallel(1, random_vec);

		array_view<float, 2> a1(50, 50, index_vec);

		array_view<float, 2> a2(50, 1, test_vec);



		array_view<float, 2> rand(50, 1, random_vec);

		parallel_for_each(
			a1.extent, [=](index<2> idx) restrict(amp)
		{
			int row = idx[0];
			int col = idx[1];

			float row_indexes_col = rand(row, 0);// (float)row / 50;

			float row_val = a2(row, 0);

			if (row == 0)
			{
				a1(row, col) = 0.0;
			}
			else
			{
				if (col > 0)
				{
					float a2_col = a2(col, 0);
					float a2_col_sub1 = a2(col - 1, 0);
					if ((a2_col >= row_indexes_col) && (a2_col_sub1 < row_indexes_col))
					{
						a1(row, col) = col;
					}
					else
					{
						a1(row, col) = 0.0;
					}
				}
				else
				{
					a1(row, col) = 0.0;

				}
			}

			//			a1(row, col) = row * 2 + col;
		});
		a1.synchronize();

		vector< vector<float> > row_vectors(50);
		for (int j = 0; j < 50; j++)
		{
			for (int i = 0; i < 50; i++)
				row_vectors[j].push_back(0.0);
		}

		for (int j = 0; j < 50; j++)
		{
			array_view<float, 2> row_a2(50, 1, row_vectors[j]);
			//CopyRowToColumn(a1, j, row_a2, 50);

			parallel_for_each(
				a1.extent, [=](index<2> idx) restrict(amp)
			{
				int row = idx[0];
				int col = idx[1];


				if (row == j)
				{
					float old_val = a1(j, col);
					row_a2(col, 1) = old_val;
				}
			});

		}
		cout << endl;
		cout << endl;



	

		for (int j = 0; j < 50; j++)
		{
			const unsigned window_width = 10;

			unsigned element_count = static_cast<unsigned>(50);
			assert(element_count != 0); // Cannot reduce an empty sequence.

			// Using array as temporary memory.
			array<float, 1> a_1(element_count, row_vectors[j].begin());

			// Takes care of the sum of tail elements.
			float tail_sum = 0.f;
			if ((element_count % window_width) != 0 && element_count > window_width)
			{
				tail_sum = std::accumulate(row_vectors[5].begin() + ((element_count - 1) / window_width) * window_width, row_vectors[5].end(), 0.f);
			}
			array_view<float, 1> av_tail_sum(1, &tail_sum);

			// Each thread reduces window_width elements.
			unsigned prev_s = element_count;
			for (unsigned s = element_count / window_width; s > 0; s /= window_width)
			{
				parallel_for_each(extent<1>(s), [=, &a_1](index<1> idx) restrict(amp)
				{
					float sum = 0.f;
					for (unsigned i = 0; i < window_width; i++)
					{
						sum += a_1[idx + i * s];
					}
					a_1[idx] = sum;

					// Reduce the tail in cases where the number of elements is not divisible.
					// Note: execution of this section may negatively affect the performance.
					// In production code the problem size passed to the reduction should
					// be a power of the window_width. Please refer to the blog post for more
					// information.
					if ((idx[0] == s - 1) && ((s % window_width) != 0) && (s > window_width))
					{
						for (unsigned i = ((s - 1) / window_width) * window_width; i < s; i++)
						{
							av_tail_sum[0] += a_1[i];
						}
					}
				});
				prev_s = s;
			}

			// Perform any remaining reduction on the CPU.
			std::vector<float> result(prev_s);
			copy(a_1.section(0, prev_s), result.begin());
			av_tail_sum.synchronize();

			row_vec[j] = std::accumulate(result.begin(), result.end(), tail_sum);

			cout << "RESULT WAS: " << row_vec[j] << endl;
			cout << endl;
	
		}


		for (int j = 0; j < 50; j++)
			cout << row_vec[j] << " ";

		cout << endl;

	}







	cout << endl;
	cout << endl;
	cout << "printing matrix thing" << endl;
	for (int i = 0; i < 50; i++)
	{
		for (int j = 0; j < 50; j++)
		{
			cout << index_vec[i * 50 + j] << " ";
		}
		cout << endl;
	}

	cout << endl;
	cout << endl;
	randomize_parallel(1, test_vec);

//	randomize_parallel(1, test_vec);
	cout << endl;
	cout << endl;
	{
	int * temp = new int[10];

	array_view<int, 2> a(5, 2, temp);

	parallel_for_each(
		a.extent, [=](index<2> idx) restrict(amp)
	{
		int row = idx[0];
		int col = idx[1];

		a(row, col) = row * 2 + col;
	});
	a.synchronize();

	for (int i = 0; i < 10; ++i)
		std::cout << temp[i] << " ";
	std::cout << "\n";
	}
	return 0;
} 

