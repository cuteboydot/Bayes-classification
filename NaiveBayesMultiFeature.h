#pragma once

#include "stdafx.h"
#include "targetver.h"

typedef struct inputdata_multi {
	int * pData;
	int nClass;
} INPUTDATA_MULTI;

class CNaiveBayesMultiFeature
{
public:
	CNaiveBayesMultiFeature(void);
	~CNaiveBayesMultiFeature(void);

	void init(int nSizeOutputPattern, int nSizeRecord, int nSizeFeature, int * pSizeFeatWords, INPUTDATA_MULTI ** ppDataList, bool bUseSmooth);
	void train();
	void classfication(INPUTDATA_MULTI * pTest);

private:
	// datas for training
	int m_nSizeOutputPattern;			// number of output pattern
	int m_nSizeFeature;					// number of feature(column)
	int m_nSizeRecord;					// number of records
	INPUTDATA_MULTI ** m_ppDataList;	// input datas of records
	bool m_bUseSmooth;

	// prob parameters
	int * m_pNumClass;					// number of records in each class
	int * m_pNumFeatWords;				// number of words in each feature	for smoothing
	int *** m_pppNumWordFeatClass;		// number of each word * Feature * each class

	double * m_pProbClass;				// 𝑷(𝒄)
	double *** m_pppProbWordFeatClass;	// 𝑷(𝒙|𝒄), sum of xi = d
};

