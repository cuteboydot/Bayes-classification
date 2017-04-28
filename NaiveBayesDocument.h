#pragma once

#include "stdafx.h"
#include "targetver.h"

typedef struct inputdata {
	int nCnt;
	int * pData;
	int nClass;
} INPUTDATA;

class CNaiveBayesDocument
{
public:
	CNaiveBayesDocument(void);
	~CNaiveBayesDocument(void);

	void init(int nSizeOutputPattern, int nSizeDocWords, int nSizeRecord, INPUTDATA ** ppDataList, bool m_bUseSmooth);
	void train();
	void classfication(INPUTDATA * pData);

private:
	// datas for training
	int m_nSizeOutputPattern;		// number of output pattern
	int m_nSizeDocWords;			// number of words in input datas

	int m_nSizeRecord;				// number of records
	INPUTDATA ** m_ppDataList;		// input datas of records
	bool m_bUseSmooth;

	// prob parameters
	int * m_pNumClass;				// number of records in each class
	int * m_pNumTotWordClass;		// number of total words in each class
	int ** m_ppNumWordClass;		// number of each word in each class

	double * m_pProbClass;			// 𝑷(𝒄)
	double ** m_ppProbWordClass;	// 𝑷(𝒙|𝒄), sum of xi = d

};

