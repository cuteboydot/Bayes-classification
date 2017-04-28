#include "NaiveBayesMultiFeature.h"

CNaiveBayesMultiFeature::CNaiveBayesMultiFeature(void)
{
	m_nSizeOutputPattern = 0;
	m_nSizeRecord = 0;
	
	m_pNumClass = 0;
	m_pNumFeatWords = 0;
	m_pppNumWordFeatClass = 0;

	m_pProbClass = 0;
	m_pppProbWordFeatClass = 0;
}

CNaiveBayesMultiFeature::~CNaiveBayesMultiFeature(void)
{
	delete[] m_pProbClass;
	for(int a=0; a<m_nSizeOutputPattern; a++) {
		if(m_pppProbWordFeatClass[a])
			delete[] m_pppProbWordFeatClass[a];
	}
	delete[] m_pppProbWordFeatClass;

	delete[] m_pNumClass;
	delete[] m_pNumFeatWords;
	for(int a=0; a<m_nSizeOutputPattern; a++) {
		for(int b=0; b<m_nSizeFeature; b++) {
			if(m_pppNumWordFeatClass[a][b])
				delete[] m_pppNumWordFeatClass[a][b];
		}
	}
	delete[] m_pppNumWordFeatClass;
}

void CNaiveBayesMultiFeature::init(int nSizeOutputPattern, int nSizeRecord, int nSizeFeature, int * pSizeFeatWords, INPUTDATA_MULTI ** ppDataList, bool bUseSmooth)
{
	// input datas
	m_nSizeOutputPattern = nSizeOutputPattern;
	m_nSizeRecord = nSizeRecord;
	m_nSizeFeature = nSizeFeature;
	m_pNumFeatWords = pSizeFeatWords;
	m_ppDataList = ppDataList;
	m_bUseSmooth = bUseSmooth;

	// internal parameters
	m_pProbClass = new double[m_nSizeOutputPattern];
	m_pppProbWordFeatClass = new double**[m_nSizeOutputPattern];
	for(int a=0; a<m_nSizeOutputPattern; a++) {
		m_pProbClass[a] = 0;

		m_pppProbWordFeatClass[a] = new double*[m_nSizeFeature];
		for(int b=0; b<m_nSizeFeature; b++) {
			m_pppProbWordFeatClass[a][b] = new double[m_pNumFeatWords[b]];
			for(int c=0; c<m_pNumFeatWords[b]; c++) {
				m_pppProbWordFeatClass[a][b][c] = 0;
			}
		}
	}

	m_pNumClass = new int[m_nSizeOutputPattern];
	m_pppNumWordFeatClass = new int**[m_nSizeOutputPattern];
	for(int a=0; a<m_nSizeOutputPattern; a++) {
		m_pNumClass[a] = 0;

		m_pppNumWordFeatClass[a] = new int*[m_nSizeFeature];
		for(int b=0; b<m_nSizeFeature; b++) {
			m_pppNumWordFeatClass[a][b] = new int[m_pNumFeatWords[b]];
			for(int c=0; c<m_pNumFeatWords[b]; c++) {
				m_pppNumWordFeatClass[a][b][c] = 0;
			}
		}
	}

}

void CNaiveBayesMultiFeature::train()
{
	// count words on each class
	for(int a=0; a<m_nSizeRecord; a++) {
		m_pNumClass[m_ppDataList[a]->nClass]++;

		for(int b=0; b<m_nSizeFeature; b++) {
			m_pppNumWordFeatClass[m_ppDataList[a]->nClass][b][m_ppDataList[a]->pData[b]]++;
		}
	}

	for(int a=0; a<m_nSizeOutputPattern; a++) {
		// get prob parameter of classes
		m_pProbClass[a] = (double)((double)m_pNumClass[a] / (double)m_nSizeRecord);
		printf("P(c%d) = %0.3f \n", a, m_pProbClass[a]);

		// get prob parameter of words including smoothing 
		for(int b=0; b<m_nSizeFeature; b++) {
			for(int c=0; c<m_pNumFeatWords[b]; c++) {
				if(!m_bUseSmooth)
					m_pppProbWordFeatClass[a][b][c] = (double)((double)m_pppNumWordFeatClass[a][b][c] / (double)m_pNumClass[a]);
				else
					m_pppProbWordFeatClass[a][b][c] = (double)((double)(m_pppNumWordFeatClass[a][b][c] + 1) / (double)(m_pNumClass[a] + m_pNumFeatWords[b]));
				printf("P(x%d%d | c%d) = %0.4f \n", b, c, a, m_pppProbWordFeatClass[a][b][c]);
			}
		}
	}
	printf("\n");
}

void CNaiveBayesMultiFeature::classfication(INPUTDATA_MULTI * pTest)
{
	double * pProbability = new double[m_nSizeOutputPattern];
	double dTemp = 0;

	for(int a=0; a<m_nSizeOutputPattern; a++) {
		pProbability[a] = 1;
		printf("\n");
		for(int b=0; b<m_nSizeFeature; b++) {
			printf("P(X%d | C%d) * ", b, a);
			pProbability[a] *= m_pppProbWordFeatClass[a][b][pTest->pData[b]];
		}
		pProbability[a] *= m_pProbClass[a];
		printf("P(C%d) = %.6f", a, pProbability[a]);

		if(dTemp < pProbability[a]) {
			dTemp = pProbability[a];
			pTest->nClass = a;
		}
	}
	printf("\n");

	delete[] pProbability;
}