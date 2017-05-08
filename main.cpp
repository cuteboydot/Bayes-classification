#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <float.h>

#include "NaiveBayesDocument.h"
#include "NaiveBayesMultiFeature.h"
#include "NaiveBayesMultiFeatureGaussian.h"

/**
cuteboydot@gmail.com

Naive Bayes Classification 

𝑪 = 𝒂𝒓𝒈𝒎𝒂𝒙 𝑷(𝒄|𝒅)
𝑪 = 𝒂𝒓𝒈𝒎𝒂𝒙( 𝑷(𝒅│𝒄)𝑷(𝒄) / 𝑷(𝒅) )
𝑪 = 𝒂𝒓𝒈𝒎𝒂𝒙 𝑷(𝒅│𝒄)𝑷(𝒄)
**/

void EX1();
void EX2();
void EX3();

int main()
{
	EX1();
	EX2();
	EX3();

	printf("Bye~~~!!! \n");
	return 0;
}


/**
EXAMPLE 1 : Movies category..

Document Words List
{fun(0), couple(1), love(2), fast(3), furious(4), shoot(5), fly(6)}

Class List
{Comedy(0), Action(1)}
}

|-------------------------------------------------------|
|Num	|Document(terms)					|Class		|
|-------|-----------------------------------|-----------|
|1		|fun, couple, love, love			|Comedy		|
|2		|fast, furious, shoot				|Action		|
|3		|couple, fly, fast, fun, fun		|Comedy		|
|4		|furious, shoot, shoot, fun			|Action		|
|5		|fly, fast, shoot, love				|Action		|
|6		|fast, furious, fun					|???		|
|-------------------------------------------------------|

𝑪 = 𝒂𝒓𝒈𝒎𝒂𝒙 𝑷(𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒔,𝒇𝒖𝒏│𝒄)𝑷(𝒄)
𝑷(𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒔,𝒇𝒖𝒏│𝒄)𝑷(𝒄) = 𝑷(𝒇𝒂𝒔𝒕│𝒄)•𝑷(𝒇𝒖𝒓𝒊𝒐𝒖𝒔│𝒄)•𝑷(𝒇𝒖𝒏|𝒄)

𝑷(𝒄): 𝑷(𝒄𝒐𝒎𝒆𝒅𝒚) = 𝟑/𝟓,		𝑷(𝒂𝒄𝒕𝒊𝒐𝒏) = 𝟐/𝟓
𝑷(𝒙|𝒄) = (𝒄𝒐𝒖𝒏𝒕(𝒙, 𝒄)) / (Ʃ𝒄𝒐𝒖𝒏𝒕(𝑿𝒊, 𝒄))

Ʃ𝒄𝒐𝒖𝒏𝒕(𝑿𝒊, 𝒄𝒐𝒎𝒆𝒅𝒚) = 𝟗
Ʃ𝒄𝒐𝒖𝒏𝒕(𝑿𝒊, 𝒂𝒄𝒕𝒊𝒐𝒏) = 𝟏𝟏

𝒄𝒐𝒖𝒏𝒕(𝒇𝒂𝒔𝒕, 𝒄𝒐𝒎𝒆𝒅𝒚)=𝟏,		𝒄𝒐𝒖𝒏𝒕(𝒇𝒂𝒔𝒕, 𝒂𝒄𝒕𝒊𝒐𝒏)=𝟐
𝒄𝒐𝒖𝒏𝒕(𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒄𝒐𝒎𝒆𝒅𝒚)=𝟎,	𝒄𝒐𝒖𝒏𝒕(𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒂𝒄𝒕𝒊𝒐𝒏)=𝟐
𝒄𝒐𝒖𝒏𝒕(𝒇𝒖𝒏, 𝒄𝒐𝒎𝒆𝒅𝒚)=𝟑,		𝒄𝒐𝒖𝒏𝒕(𝒇𝒖𝒏, 𝒂𝒄𝒕𝒊𝒐𝒏)=𝟏

𝑷(𝒄𝒐𝒎𝒆𝒅𝒚│𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = 𝑷(𝒇𝒂𝒔𝒕│𝒄𝒐𝒎𝒆𝒅𝒚)•𝑷(𝒇𝒖𝒓𝒊𝒐𝒖𝒔│𝒄𝒐𝒎𝒆𝒅𝒚)•𝑷(𝒇𝒖𝒏|𝒄𝒐𝒎𝒆𝒅𝒚)•𝑷(𝒄𝒐𝒎𝒆𝒅𝒚)
𝑷(𝒄𝒐𝒎𝒆𝒅𝒚|𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = 𝟏/𝟗 • 𝟎/𝟗 • 𝟑/𝟗 • 𝟐/𝟓 = 𝟎

𝑷(𝒂𝒄𝒕𝒊𝒐𝒏│𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = 𝑷(𝒇𝒂𝒔𝒕│𝒂𝒄𝒕𝒊𝒐𝒏)•𝑷(𝒇𝒖𝒓𝒊𝒐𝒖𝒔│𝒂𝒄𝒕𝒊𝒐𝒏)•𝑷(𝒇𝒖𝒏|𝒂𝒄𝒕𝒊𝒐𝒏)•𝑷(𝒂𝒄𝒕𝒊𝒐𝒏)
𝑷(𝒂𝒄𝒕𝒊𝒐𝒏|𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = 𝟐/𝟏𝟏 • 𝟐/𝟏𝟏 • 𝟏/𝟏𝟏 • 𝟑/𝟓 = 𝟎.𝟎𝟎𝟏𝟖

After Smoothing
𝑷(𝒄𝒐𝒎𝒆𝒅𝒚|𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = (𝟏+𝟏)/(𝟗+𝟕) • (𝟎+𝟏)/(𝟗+𝟕) • (𝟑+𝟏)/(𝟗+𝟕) • 𝟐/𝟓 = 𝟎.𝟎𝟎𝟎𝟕𝟖
𝑷(𝒂𝒄𝒕𝒊𝒐𝒏|𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = (𝟐+𝟏)/(𝟏𝟏+𝟕) • (𝟐+𝟏)/(𝟏𝟏+𝟕) • (𝟏+𝟏)/(𝟏𝟏+𝟕) • 𝟑/𝟓 = 𝟎.𝟎𝟎𝟏𝟖
**/
void EX1()
{
#define SIZE_RECORD		5
#define SIZE_OUTPUT		2
#define SIZE_WORDLIST	7

	enum WORDLIST {FUN=0, COUPLE, LOVE, FAST, FURIOUS, SHOOT, FLY};
	enum ANSWERLIST {COMEDY=0, ACTION};

	INPUTDATA ** ppInputData;
	INPUTDATA * pTestData;
	
	ppInputData = new INPUTDATA*[SIZE_RECORD];
	pTestData = new INPUTDATA;

	ppInputData[0] = new INPUTDATA;
	ppInputData[0]->nCnt = 4;
	ppInputData[0]->pData = new int[ppInputData[0]->nCnt];
	ppInputData[0]->pData[0] = FUN;
	ppInputData[0]->pData[1] = COUPLE;
	ppInputData[0]->pData[2] = LOVE;
	ppInputData[0]->pData[3] = LOVE;
	ppInputData[0]->nClass = COMEDY;
	
	ppInputData[1] = new INPUTDATA;
	ppInputData[1]->nCnt = 3;
	ppInputData[1]->pData = new int[ppInputData[1]->nCnt];
	ppInputData[1]->pData[0] = FAST;
	ppInputData[1]->pData[1] = FURIOUS;
	ppInputData[1]->pData[2] = SHOOT;
	ppInputData[1]->nClass = ACTION;

	ppInputData[2] = new INPUTDATA;
	ppInputData[2]->nCnt = 5;
	ppInputData[2]->pData = new int[ppInputData[2]->nCnt];
	ppInputData[2]->pData[0] = COUPLE;
	ppInputData[2]->pData[1] = FLY;
	ppInputData[2]->pData[2] = FAST;
	ppInputData[2]->pData[3] = FUN;
	ppInputData[2]->pData[4] = FUN;
	ppInputData[2]->nClass = COMEDY;

	ppInputData[3] = new INPUTDATA;
	ppInputData[3]->nCnt = 4;
	ppInputData[3]->pData = new int[ppInputData[3]->nCnt];
	ppInputData[3]->pData[0] = FURIOUS;
	ppInputData[3]->pData[1] = SHOOT;
	ppInputData[3]->pData[2] = SHOOT;
	ppInputData[3]->pData[3] = FUN;
	ppInputData[3]->nClass = ACTION;

	ppInputData[4] = new INPUTDATA;
	ppInputData[4]->nCnt = 4;
	ppInputData[4]->pData = new int[ppInputData[4]->nCnt];
	ppInputData[4]->pData[0] = FLY;
	ppInputData[4]->pData[1] = FAST;
	ppInputData[4]->pData[2] = SHOOT;
	ppInputData[4]->pData[3] = LOVE;
	ppInputData[4]->nClass = ACTION;

	pTestData->nCnt = 3;
	pTestData->pData = new int[pTestData->nCnt];
	pTestData->pData[0] = FAST;
	pTestData->pData[1] = FURIOUS;
	pTestData->pData[2] = FUN;
	pTestData->nClass = -1;

	printf("----------------------EXAMPLE#1----------------------\n");
	CNaiveBayesDocument * pNaiveBayes = new CNaiveBayesDocument();
	pNaiveBayes->init(SIZE_OUTPUT, SIZE_WORDLIST, SIZE_RECORD, ppInputData, true);
	pNaiveBayes->train();
	pNaiveBayes->classfication(pTestData);
	printf("-----------------------------------------------------\n\n");

	// terminate memory	
	for(int a=0; a<SIZE_RECORD; a++) {
		if(ppInputData[a]->pData) {
			delete[] ppInputData[a]->pData;
			delete ppInputData[a];
		}
	}
	delete[] ppInputData;

	if(pTestData) {
		delete[] pTestData->pData;
		delete[] pTestData;
	}

#undef SIZE_RECORD		
#undef SIZE_OUTPUT		
#undef SIZE_WORDLIST
}


/**
EXAMPLE 2 : Playing tennis..
|---------------------------------------------------------------|
|Num	|Outlook	|Temperature	|Humidity	|Wind	|Class	|
|-------|-----------|---------------|-----------|-------|-------|
|1		|Sunny		|Hot			|High		|Weak	|No		|
|2		|Sunny		|Hot			|High		|Strong	|No		|
|3		|Overcast	|Hot			|High		|Weak	|Yes	|
|4		|Rain		|Mild			|High		|Weak	|Yes	|
|5		|Rain		|Cool			|Normal		|Weak	|Yes	|
|6		|Rain		|Cool			|Normal		|Strong	|No		|
|7		|Overcast	|Cool			|Normal		|Strong	|Yes	|
|8		|Sunny		|Mild			|High		|Weak	|No		|
|9		|Sunny		|Cool			|Normal		|Weak	|Yes	|
|10		|Rain		|Mild			|Normal		|Weak	|Yes	|
|11		|Sunny		|Mild			|Normal		|Strong	|Yes	|
|12		|Overcast	|Mild			|High		|Strong	|Yes	|
|13		|Overcast	|Hot			|Normal		|Weak	|Yes	|
|14		|Rain		|Mild			|High		|Strong	|No		|
|15		|Sunny		|Cool			|High		|Strong	|???	|
|---------------------------------------------------------------|

𝑷(𝒚𝒆𝒔)=𝟗/𝟏𝟒,  𝑷(𝒏𝒐)=𝟓/𝟏𝟒
𝑷(𝒘𝒊𝒏𝒅=𝒔𝒕𝒓𝒐𝒏𝒈|𝒚𝒆𝒔)=𝟑/𝟗,  𝑷(𝒘𝒊𝒏𝒅=𝒔𝒕𝒓𝒐𝒏𝒈|𝒏𝒐)=𝟑/𝟓
...
𝑷(𝒚)𝑷(𝒔𝒖𝒏│𝒚)𝑷(𝒄𝒐𝒐𝒍│𝒚)𝑷(𝒉𝒊𝒈𝒉│𝒚)𝑷(𝒔𝒕𝒓𝒐𝒏𝒈│𝒚) = 𝟎.𝟎𝟎𝟓
𝑷(𝒏)𝑷(𝒔𝒖𝒏│𝒏)𝑷(𝒄𝒐𝒐𝒍│𝒏)𝑷(𝒉𝒊𝒈𝒉│𝒏)𝑷(𝒔𝒕𝒓𝒐𝒏𝒈│𝒏) = 𝟎.𝟎𝟐𝟏
**/
void EX2()
{
#define SIZE_RECORD		14
#define SIZE_OUTPUT		2
#define SIZE_FEATURE	4

	enum WORDLIST_OUTLOOK		{SUNNY = 0, OVERCAST, RAIN};
	enum WORDLIST_TEMPERATURE	{HOT = 0, MILD, COOL};
	enum WORDLIST_HUMIDITY		{HIGH = 0, NORMAL};
	enum WORDLIST_WIND			{WEAK = 0, STRONG};
	enum ANSWERLIST				{NO = 0, YES};

	INPUTDATA_MULTI ** ppInputData;
	INPUTDATA_MULTI * pTestData;
	int * pFeatWords;

	ppInputData = new INPUTDATA_MULTI*[SIZE_RECORD];
	pTestData = new INPUTDATA_MULTI;

	pFeatWords = new int[SIZE_RECORD];
	pFeatWords[0] = 3;	// sizeof(WORDLIST_OUTLOOK)
	pFeatWords[1] = 3;	// sizeof(WORDLIST_TEMPERATURE)
	pFeatWords[2] = 2;	// sizeof(WORDLIST_HUMIDITY)
	pFeatWords[3] = 2;	// sizeof(WORDLIST_WIND)

	ppInputData[0] = new INPUTDATA_MULTI;
	ppInputData[0]->pData = new int[SIZE_FEATURE];
	ppInputData[0]->pData[0] = SUNNY;
	ppInputData[0]->pData[1] = HOT;
	ppInputData[0]->pData[2] = HIGH;
	ppInputData[0]->pData[3] = WEAK;
	ppInputData[0]->nClass = NO;

	ppInputData[1] = new INPUTDATA_MULTI;
	ppInputData[1]->pData = new int[SIZE_FEATURE];
	ppInputData[1]->pData[0] = SUNNY;
	ppInputData[1]->pData[1] = HOT;
	ppInputData[1]->pData[2] = HIGH;
	ppInputData[1]->pData[3] = STRONG;
	ppInputData[1]->nClass = NO;

	ppInputData[2] = new INPUTDATA_MULTI;
	ppInputData[2]->pData = new int[SIZE_FEATURE];
	ppInputData[2]->pData[0] = OVERCAST;
	ppInputData[2]->pData[1] = HOT;
	ppInputData[2]->pData[2] = HIGH;
	ppInputData[2]->pData[3] = WEAK;
	ppInputData[2]->nClass = YES;

	ppInputData[3] = new INPUTDATA_MULTI;
	ppInputData[3]->pData = new int[SIZE_FEATURE];
	ppInputData[3]->pData[0] = RAIN;
	ppInputData[3]->pData[1] = MILD;
	ppInputData[3]->pData[2] = HIGH;
	ppInputData[3]->pData[3] = WEAK;
	ppInputData[3]->nClass = YES;

	ppInputData[4] = new INPUTDATA_MULTI;
	ppInputData[4]->pData = new int[SIZE_FEATURE];
	ppInputData[4]->pData[0] = RAIN;
	ppInputData[4]->pData[1] = COOL;
	ppInputData[4]->pData[2] = NORMAL;
	ppInputData[4]->pData[3] = WEAK;
	ppInputData[4]->nClass = YES;

	ppInputData[5] = new INPUTDATA_MULTI;
	ppInputData[5]->pData = new int[SIZE_FEATURE];
	ppInputData[5]->pData[0] = RAIN;
	ppInputData[5]->pData[1] = COOL;
	ppInputData[5]->pData[2] = NORMAL;
	ppInputData[5]->pData[3] = STRONG;
	ppInputData[5]->nClass = NO;

	ppInputData[6] = new INPUTDATA_MULTI;
	ppInputData[6]->pData = new int[SIZE_FEATURE];
	ppInputData[6]->pData[0] = OVERCAST;
	ppInputData[6]->pData[1] = COOL;
	ppInputData[6]->pData[2] = NORMAL;
	ppInputData[6]->pData[3] = STRONG;
	ppInputData[6]->nClass = YES;

	ppInputData[7] = new INPUTDATA_MULTI;
	ppInputData[7]->pData = new int[SIZE_FEATURE];
	ppInputData[7]->pData[0] = SUNNY;
	ppInputData[7]->pData[1] = MILD;
	ppInputData[7]->pData[2] = HIGH;
	ppInputData[7]->pData[3] = WEAK;
	ppInputData[7]->nClass = NO;

	ppInputData[8] = new INPUTDATA_MULTI;
	ppInputData[8]->pData = new int[SIZE_FEATURE];
	ppInputData[8]->pData[0] = SUNNY;
	ppInputData[8]->pData[1] = COOL;
	ppInputData[8]->pData[2] = NORMAL;
	ppInputData[8]->pData[3] = WEAK;
	ppInputData[8]->nClass = YES;

	ppInputData[9] = new INPUTDATA_MULTI;
	ppInputData[9]->pData = new int[SIZE_FEATURE];
	ppInputData[9]->pData[0] = RAIN;
	ppInputData[9]->pData[1] = MILD;
	ppInputData[9]->pData[2] = NORMAL;
	ppInputData[9]->pData[3] = WEAK;
	ppInputData[9]->nClass = YES;

	ppInputData[10] = new INPUTDATA_MULTI;
	ppInputData[10]->pData = new int[SIZE_FEATURE];
	ppInputData[10]->pData[0] = SUNNY;
	ppInputData[10]->pData[1] = MILD;
	ppInputData[10]->pData[2] = NORMAL;
	ppInputData[10]->pData[3] = STRONG;
	ppInputData[10]->nClass = YES;

	ppInputData[11] = new INPUTDATA_MULTI;
	ppInputData[11]->pData = new int[SIZE_FEATURE];
	ppInputData[11]->pData[0] = OVERCAST;
	ppInputData[11]->pData[1] = MILD;
	ppInputData[11]->pData[2] = HIGH;
	ppInputData[11]->pData[3] = STRONG;
	ppInputData[11]->nClass = YES;

	ppInputData[12] = new INPUTDATA_MULTI;
	ppInputData[12]->pData = new int[SIZE_FEATURE];
	ppInputData[12]->pData[0] = OVERCAST;
	ppInputData[12]->pData[1] = HOT;
	ppInputData[12]->pData[2] = NORMAL;
	ppInputData[12]->pData[3] = WEAK;
	ppInputData[12]->nClass = YES;

	ppInputData[13] = new INPUTDATA_MULTI;
	ppInputData[13]->pData = new int[SIZE_FEATURE];
	ppInputData[13]->pData[0] = RAIN;
	ppInputData[13]->pData[1] = MILD;
	ppInputData[13]->pData[2] = HIGH;
	ppInputData[13]->pData[3] = STRONG;
	ppInputData[13]->nClass = NO;

	pTestData->pData = new int[SIZE_FEATURE];
	pTestData->pData[0] = SUNNY;
	pTestData->pData[1] = COOL;
	pTestData->pData[2] = HIGH;
	pTestData->pData[3] = STRONG;
	pTestData->nClass = -1;

	printf("----------------------EXAMPLE#2----------------------\n");
	CNaiveBayesMultiFeature * pNaiveBayesMulti = new CNaiveBayesMultiFeature();
	pNaiveBayesMulti->init(SIZE_OUTPUT, SIZE_RECORD, SIZE_FEATURE, pFeatWords, ppInputData, true);
	pNaiveBayesMulti->train();
	pNaiveBayesMulti->classfication(pTestData);
	printf("-----------------------------------------------------\n\n");

	// terminate memory	
	for(int a=0; a<SIZE_RECORD; a++) {
		if(ppInputData[a]) {
			delete[] ppInputData[a]->pData;
			delete[] ppInputData[a];
		}
	}
	delete[] ppInputData;

	if(pTestData) {
		delete[] pTestData->pData;
		delete[] pTestData;
	}

#undef SIZE_RECORD		
#undef SIZE_OUTPUT		
#undef SIZE_FEATURE

}

/**
EXAMPLE 3 : Male or female..
|---------------------------------------|
|Num	|Height	|Weight	|foot	|Class	|
|-------|-------|-------|-------|-------|
|1		|6		|180	|12		|Male	|
|2		|5.92	|190	|11		|Male	|
|3		|5.58	|170	|12		|Male	|
|4		|5.92	|165	|10		|Male	|
|5		|5		|100	|6		|Female	|
|6		|5.5	|150	|8		|Female	|
|7		|5.42	|130	|7		|Female	|
|8		|5.75	|150	|9		|Female	|
|9		|6		|130	|8		|???	|
|---------------------------------------|
**/
void EX3()
{
#define SIZE_RECORD		8
#define SIZE_OUTPUT		2
#define SIZE_FEATURE	3

	enum ANSWERLIST	{MALE=0, FEMALE};

	INPUTDATA_MULTI_GAUSS ** ppInputData;
	INPUTDATA_MULTI_GAUSS * pTestData;

	ppInputData = new INPUTDATA_MULTI_GAUSS*[SIZE_RECORD];
	pTestData = new INPUTDATA_MULTI_GAUSS;

	ppInputData[0] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[0]->pData = new double[SIZE_FEATURE];
	ppInputData[0]->pData[0] = 6;
	ppInputData[0]->pData[1] = 180;
	ppInputData[0]->pData[2] = 12;
	ppInputData[0]->nClass = MALE;

	ppInputData[1] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[1]->pData = new double[SIZE_FEATURE];
	ppInputData[1]->pData[0] = 5.92;
	ppInputData[1]->pData[1] = 190;
	ppInputData[1]->pData[2] = 11;
	ppInputData[1]->nClass = MALE;

	ppInputData[2] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[2]->pData = new double[SIZE_FEATURE];
	ppInputData[2]->pData[0] = 5.58;
	ppInputData[2]->pData[1] = 170;
	ppInputData[2]->pData[2] = 12;
	ppInputData[2]->nClass = MALE;

	ppInputData[3] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[3]->pData = new double[SIZE_FEATURE];
	ppInputData[3]->pData[0] = 5.92;
	ppInputData[3]->pData[1] = 165;
	ppInputData[3]->pData[2] = 10;
	ppInputData[3]->nClass = MALE;

	ppInputData[4] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[4]->pData = new double[SIZE_FEATURE];
	ppInputData[4]->pData[0] = 5;
	ppInputData[4]->pData[1] = 100;
	ppInputData[4]->pData[2] = 6;
	ppInputData[4]->nClass = FEMALE;

	ppInputData[5] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[5]->pData = new double[SIZE_FEATURE];
	ppInputData[5]->pData[0] = 5.5;
	ppInputData[5]->pData[1] = 150;
	ppInputData[5]->pData[2] = 8;
	ppInputData[5]->nClass = FEMALE;

	ppInputData[6] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[6]->pData = new double[SIZE_FEATURE];
	ppInputData[6]->pData[0] = 5.42;
	ppInputData[6]->pData[1] = 130;
	ppInputData[6]->pData[2] = 7;
	ppInputData[6]->nClass = FEMALE;

	ppInputData[7] = new INPUTDATA_MULTI_GAUSS;
	ppInputData[7]->pData = new double[SIZE_FEATURE];
	ppInputData[7]->pData[0] = 5.75;
	ppInputData[7]->pData[1] = 150;
	ppInputData[7]->pData[2] = 9;
	ppInputData[7]->nClass = FEMALE;

	pTestData = new INPUTDATA_MULTI_GAUSS;
	pTestData->pData = new double[SIZE_FEATURE];
	pTestData->pData[0] = 6;
	pTestData->pData[1] = 130;
	pTestData->pData[2] = 8;
	pTestData->nClass = -1;

	printf("----------------------EXAMPLE#3----------------------\n");
	CNaiveBayesMultiFeatureGaussian * pNaiveBayesMultiGauss = new CNaiveBayesMultiFeatureGaussian();
	pNaiveBayesMultiGauss->init(SIZE_OUTPUT, SIZE_RECORD, SIZE_FEATURE, ppInputData);
	pNaiveBayesMultiGauss->train();
	pNaiveBayesMultiGauss->classfication(pTestData, false);
	printf("-----------------------------------------------------\n\n");

	// terminate memory	
	for(int a=0; a<SIZE_RECORD; a++) {
		if(ppInputData[a]) {
			delete[] ppInputData[a]->pData;
			delete[] ppInputData[a];
		}
	}
	delete[] ppInputData;

	if(pTestData) {
		delete[] pTestData->pData;
		delete[] pTestData;
	}

#undef SIZE_RECORD		
#undef SIZE_OUTPUT		
#undef SIZE_FEATURE
}
