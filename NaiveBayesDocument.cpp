#include "NaiveBayesDocument.h"

CNaiveBayesDocument::CNaiveBayesDocument(void)
{
	m_nSizeOutputPattern = 0;
	m_nSizeDocWords = 0;

	m_nSizeRecord = 0;
	m_ppDataList = 0;
}

CNaiveBayesDocument::~CNaiveBayesDocument(void)
{
	delete[] m_pProbClass;
	for(int a=0; a<m_nSizeOutputPattern; a++) {
		if(m_ppProbWordClass[a]) 
			delete[] m_ppProbWordClass[a];
	}
	delete[] m_ppProbWordClass;

	delete[] m_pNumClass;
	delete[] m_pNumTotWordClass;
	for(int a=0; a<m_nSizeOutputPattern; a++) {
		if(m_ppNumWordClass[a]) 
			delete[] m_ppNumWordClass[a];
	}
	delete[] m_ppNumWordClass;
}

// init datas
void CNaiveBayesDocument::init(int nSizeOutputPattern, int nSizeDocWords, int nSizeRecord, INPUTDATA ** ppDataList, bool UseSmooth)
{
	// input datas
	m_nSizeOutputPattern = nSizeOutputPattern;
	m_nSizeDocWords = nSizeDocWords;
	m_nSizeRecord = nSizeRecord;
	m_ppDataList = ppDataList;
    m_bUseSmooth = UseSmooth;

	// internal parameters
	m_pProbClass = new double[m_nSizeOutputPattern];
	m_ppProbWordClass = new double*[m_nSizeOutputPattern];
	for(int a=0; a<m_nSizeOutputPattern; a++) {
		m_pProbClass[a] = 0;
		m_ppProbWordClass[a] = new double[m_nSizeDocWords];
		for(int b=0; b<m_nSizeDocWords; b++) {
			m_ppProbWordClass[a][b] = 0;
		}
	}

	m_pNumClass = new int[m_nSizeOutputPattern];
	m_pNumTotWordClass = new int[m_nSizeOutputPattern];
	m_ppNumWordClass = new int*[m_nSizeOutputPattern];
	for(int a=0; a<m_nSizeOutputPattern; a++) {
		m_pNumClass[a] = 0;
		m_pNumTotWordClass[a] = 0;
		m_ppNumWordClass[a] = new int[m_nSizeDocWords];
		for(int b=0; b<m_nSizeDocWords; b++) {
			m_ppNumWordClass[a][b] = 0;
		}
	}
}

/**
calc parameters
P(Ci)			: m_pProbClass
Ʃ𝒄𝒐𝒖𝒏𝒕(Xj, Ci)	: m_ppNumWordClass
𝑷(Xj|Ci)		: m_ppProbWordClass
**/
void CNaiveBayesDocument::train()
{
	// count words on each class
	for(int a=0; a<m_nSizeRecord; a++) {
		m_pNumClass[m_ppDataList[a]->nClass]++;
		
		m_pNumTotWordClass[m_ppDataList[a]->nClass] += m_ppDataList[a]->nCnt;
		for(int b=0; b<m_ppDataList[a]->nCnt; b++) {
			m_ppNumWordClass[m_ppDataList[a]->nClass][m_ppDataList[a]->pData[b]]++;
		}
	}

	for(int a=0; a<m_nSizeOutputPattern; a++) {
		// get prob parameter of classes
		m_pProbClass[a] = (double)((double)m_pNumClass[a] / (double)m_nSizeRecord);
		printf("P(c%d) = %0.3f \n", a, m_pProbClass[a]);

		// get prob parameter of words including smoothing 
		for(int b=0; b<m_nSizeDocWords; b++) {
			if(!m_bUseSmooth)
                m_ppProbWordClass[a][b] = (double)((double)m_ppNumWordClass[a][b] / (double)m_pNumTotWordClass[a]);
			else
                m_ppProbWordClass[a][b] = (double)((double)(m_ppNumWordClass[a][b] + 1) / (double)(m_pNumTotWordClass[a] + m_nSizeDocWords));
			printf("P(x%d | c%d) = %0.4f \n", b, a, m_ppProbWordClass[a][b]);
		}
		printf("\n");
	}
}

/**
testing func

compare with
P(X1|C1)*P(X2|C1)...P(Xn|C1)*P(C1)
and
P(X1|C2)*P(X2|C2)...P(Xn|C2)*P(C2)
**/
void CNaiveBayesDocument::classfication(INPUTDATA * pData)
{
	double * pProbability = new double[m_nSizeOutputPattern];
	double dTemp = 0;

	for(int a=0; a<m_nSizeOutputPattern; a++) {
		pProbability[a] = 1;
		printf("\n");
		for(int b=0; b<pData->nCnt; b++) {
			printf("P(X%d|C%d) * ", pData->pData[b], a);
			pProbability[a] *= m_ppProbWordClass[a][pData->pData[b]];
		}
		pProbability[a] *= m_pProbClass[a];
		printf("P(C%d) = %.8f", a, pProbability[a]);

		if(dTemp < pProbability[a]) {
			dTemp = pProbability[a];
			pData->nClass = a;
		}
	}
	printf("\n");

	delete[] pProbability;
}

