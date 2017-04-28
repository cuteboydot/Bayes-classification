#include "NaiveBayesMultiFeatureGaussian.h"


CNaiveBayesMultiFeatureGaussian::CNaiveBayesMultiFeatureGaussian(void)
{
	m_nSizeOutputPattern = 0;
	m_nSizeRecord = 0;
	m_ppDataList = 0;

	m_pNumClass = 0;
	m_pProbClass = 0;
	m_ppMeanFeatClass = 0;
	m_ppVarFeatClass = 0;
	m_ppSumFeatClass = 0;
	m_ppSumVarClass = 0;
}

CNaiveBayesMultiFeatureGaussian::~CNaiveBayesMultiFeatureGaussian(void)
{
	delete[] m_pNumClass;
	delete[] m_pProbClass;

	for(int a=0; a<m_nSizeOutputPattern; a++) {
		if(m_ppMeanFeatClass[a]) 
			delete[] m_ppMeanFeatClass[a];
		if(m_ppVarFeatClass[a]) 
			delete[] m_ppVarFeatClass[a];
		if(m_ppSumFeatClass[a])
			delete[] m_ppSumFeatClass[a];
		if(m_ppSumVarClass[a])
			delete[] m_ppSumVarClass[a];
	}
	delete[] m_ppMeanFeatClass;
	delete[] m_ppVarFeatClass;
	delete[] m_ppSumFeatClass;
	delete[] m_ppSumVarClass;
}

void CNaiveBayesMultiFeatureGaussian::init(int nSizeOutputPattern, int nSizeRecord, int nSizeFeature, INPUTDATA_MULTI_GAUSS ** ppDataList)
{
	// input datas
	m_nSizeOutputPattern = nSizeOutputPattern;
	m_nSizeRecord = nSizeRecord;
	m_nSizeFeature = nSizeFeature;
	m_ppDataList = ppDataList;

	// internal parameters
	m_pNumClass = new int[m_nSizeOutputPattern];
	m_pProbClass = new double[m_nSizeOutputPattern];

	m_ppSumFeatClass = new double*[m_nSizeOutputPattern];
	m_ppSumVarClass = new double*[m_nSizeOutputPattern];
	m_ppMeanFeatClass = new double*[m_nSizeOutputPattern];
	m_ppVarFeatClass = new double*[m_nSizeOutputPattern];
	for(int a=0; a<m_nSizeOutputPattern; a++) {
		m_pNumClass[a] = 0;
		m_pProbClass[a] = 0;

		m_ppSumFeatClass[a] = new double[m_nSizeFeature];
		m_ppSumVarClass[a] = new double[m_nSizeFeature];
		m_ppMeanFeatClass[a] = new double[m_nSizeFeature];
		m_ppVarFeatClass[a] = new double[m_nSizeFeature];
		for(int b=0; b<m_nSizeFeature; b++) {
			m_ppSumFeatClass[a][b] = 0;
			m_ppSumVarClass[a][b] = 0;
			m_ppMeanFeatClass[a][b] = 0;
			m_ppVarFeatClass[a][b] = 0;
		}
	}
}

void CNaiveBayesMultiFeatureGaussian::train()
{
	// count & sum values
	for(int a=0; a<m_nSizeRecord; a++) {
		m_pNumClass[m_ppDataList[a]->nClass]++;

		for(int b=0; b<m_nSizeFeature; b++) {
			m_ppSumFeatClass[m_ppDataList[a]->nClass][b] += m_ppDataList[a]->pData[b]; 
		}
	}

	// calc mean
	for(int a=0; a<m_nSizeOutputPattern; a++) {
		m_pProbClass[a] = (double)((double)m_pNumClass[a] / (double)m_nSizeRecord);

		for(int b=0; b<m_nSizeFeature; b++) {
			m_ppMeanFeatClass[a][b] = (double)((double)m_ppSumFeatClass[a][b] / (double)m_pNumClass[a]);
		}
	}

	// calc variance
	for(int a=0; a<m_nSizeRecord; a++) {
		for(int b=0; b<m_nSizeFeature; b++) {
			m_ppSumVarClass[m_ppDataList[a]->nClass][b] = m_ppSumVarClass[m_ppDataList[a]->nClass][b] + 
				(m_ppDataList[a]->pData[b] - m_ppMeanFeatClass[m_ppDataList[a]->nClass][b]) * 
				(m_ppDataList[a]->pData[b] - m_ppMeanFeatClass[m_ppDataList[a]->nClass][b]);
		} 
	}

	for(int a=0; a<m_nSizeOutputPattern; a++) {
		printf("P(c%d) = %0.3f \n", a, m_pProbClass[a]);
		for(int b=0; b<m_nSizeFeature; b++) {
			m_ppVarFeatClass[a][b] = m_ppSumVarClass[a][b] / (double)(m_pNumClass[a] - 1);
			printf("Mean[%d][%d]=%.4f,\tVariance[%d][%d]=%.4f \n", a, b, m_ppMeanFeatClass[a][b], a, b, m_ppVarFeatClass[a][b]);
		}
	}
}

double CNaiveBayesMultiFeatureGaussian::getgauss(double dMean, double dVar, double dValue)
{
	double dGauss = 1;
	const double dPi = 3.14159265358979323846;

	dGauss = (1 / sqrt(2 * dPi * dVar)) * (exp((-1 * (dValue - dMean) * (dValue - dMean)) / (2*dVar)));

	return dGauss;
}

void CNaiveBayesMultiFeatureGaussian::classfication(INPUTDATA_MULTI_GAUSS * pTest, bool bUseLog)
{
	double * pProbability = new double[m_nSizeOutputPattern];
	double dGauss = 1;
	double dTemp = 0;

	for(int a=0; a<m_nSizeOutputPattern; a++) {
		pProbability[a] = 1;
		if(bUseLog)
			pProbability[a] = 0;
		dGauss = 1;

		printf("\n");
		for(int b=0; b<m_nSizeFeature; b++) {
			printf("P(X%d | C%d) * ", b, a);
			dGauss = getgauss(m_ppMeanFeatClass[a][b], m_ppVarFeatClass[a][b], pTest->pData[b]);

			if (bUseLog) pProbability[a] += log(dGauss);
			else pProbability[a] *= dGauss;
		}

		if (bUseLog) pProbability[a] += log(dGauss);
		else pProbability[a] *= dGauss;

		printf("P(C%d) = %.12f", a, pProbability[a]);

		if(dTemp < pProbability[a]) {
			dTemp = pProbability[a];
			pTest->nClass = a;
		}
	}
	printf("\n");

	delete[] pProbability;
}
