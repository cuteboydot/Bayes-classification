# Bayes-classification
Implemation of naive bayes classification

cuteboydot@gmail.com

𝑪 = 𝒂𝒓𝒈𝒎𝒂𝒙 𝑷(𝒄|𝒅)
𝑪 = 𝒂𝒓𝒈𝒎𝒂𝒙( 𝑷(𝒅│𝒄)𝑷(𝒄) / 𝑷(𝒅) )
𝑪 = 𝒂𝒓𝒈𝒎𝒂𝒙 𝑷(𝒅│𝒄)𝑷(𝒄)


EXAMPLE 1 : Movies category..
|-------------------------------------------------------|
|Num    |Document(terms)                    |Class      |
|-------|-----------------------------------|-----------|
|1      |fun, couple, love, love            |Comedy     |
|2      |fast, furious, shoot               |Action     |
|3      |couple, fly, fast, fun, fun        |Comedy     |
|4      |furious, shoot, shoot, fun         |Action     |
|5      |fly, fast, shoot, love             |Action     |
|6      |fast, furious, fun                 |???        |
|-------------------------------------------------------|
Document Words List = {fun(0), couple(1), love(2), fast(3), furious(4), shoot(5), fly(6)}
Class List = {Comedy(0), Action(1)}}

𝑪 = 𝒂𝒓𝒈𝒎𝒂𝒙 𝑷(𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒔,𝒇𝒖𝒏│𝒄)𝑷(𝒄)
𝑷(𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒔,𝒇𝒖𝒏│𝒄)𝑷(𝒄) = 𝑷(𝒇𝒂𝒔𝒕│𝒄)•𝑷(𝒇𝒖𝒓𝒊𝒐𝒖𝒔│𝒄)•𝑷(𝒇𝒖𝒏|𝒄)

𝑷(𝒄): 𝑷(𝒄𝒐𝒎𝒆𝒅𝒚) = 𝟑/𝟓,  𝑷(𝒂𝒄𝒕𝒊𝒐𝒏) = 𝟐/𝟓
𝑷(𝒙|𝒄) = (𝒄𝒐𝒖𝒏𝒕(𝒙, 𝒄)) / (Ʃ𝒄𝒐𝒖𝒏𝒕(𝑿𝒊, 𝒄))

Ʃ𝒄𝒐𝒖𝒏𝒕(𝑿𝒊, 𝒄𝒐𝒎𝒆𝒅𝒚) = 𝟗
Ʃ𝒄𝒐𝒖𝒏𝒕(𝑿𝒊, 𝒂𝒄𝒕𝒊𝒐𝒏) = 𝟏𝟏

𝒄𝒐𝒖𝒏𝒕(𝒇𝒂𝒔𝒕, 𝒄𝒐𝒎𝒆𝒅𝒚)=𝟏,  𝒄𝒐𝒖𝒏𝒕(𝒇𝒂𝒔𝒕, 𝒂𝒄𝒕𝒊𝒐𝒏)=𝟐
𝒄𝒐𝒖𝒏𝒕(𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒄𝒐𝒎𝒆𝒅𝒚)=𝟎, 𝒄𝒐𝒖𝒏𝒕(𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒂𝒄𝒕𝒊𝒐𝒏)=𝟐
𝒄𝒐𝒖𝒏𝒕(𝒇𝒖𝒏, 𝒄𝒐𝒎𝒆𝒅𝒚)=𝟑,  𝒄𝒐𝒖𝒏𝒕(𝒇𝒖𝒏, 𝒂𝒄𝒕𝒊𝒐𝒏)=𝟏

𝑷(𝒄𝒐𝒎𝒆𝒅𝒚│𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = 𝑷(𝒇𝒂𝒔𝒕│𝒄𝒐𝒎𝒆𝒅𝒚)•𝑷(𝒇𝒖𝒓𝒊𝒐𝒖𝒔│𝒄𝒐𝒎𝒆𝒅𝒚)•𝑷(𝒇𝒖𝒏|𝒄𝒐𝒎𝒆𝒅𝒚)•𝑷(𝒄𝒐𝒎𝒆𝒅𝒚)
𝑷(𝒄𝒐𝒎𝒆𝒅𝒚|𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = 𝟏/𝟗 • 𝟎/𝟗 • 𝟑/𝟗 • 𝟐/𝟓 = 𝟎
𝑷(𝒂𝒄𝒕𝒊𝒐𝒏│𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = 𝑷(𝒇𝒂𝒔𝒕│𝒂𝒄𝒕𝒊𝒐𝒏)•𝑷(𝒇𝒖𝒓𝒊𝒐𝒖𝒔│𝒂𝒄𝒕𝒊𝒐𝒏)•𝑷(𝒇𝒖𝒏|𝒂𝒄𝒕𝒊𝒐𝒏)•𝑷(𝒂𝒄𝒕𝒊𝒐𝒏)
𝑷(𝒂𝒄𝒕𝒊𝒐𝒏|𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = 𝟐/𝟏𝟏 • 𝟐/𝟏𝟏 • 𝟏/𝟏𝟏 • 𝟑/𝟓 = 𝟎.𝟎𝟎𝟏𝟖

After Smoothing
𝑷(𝒄𝒐𝒎𝒆𝒅𝒚|𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = (𝟏+𝟏)/(𝟗+𝟕) • (𝟎+𝟏)/(𝟗+𝟕) • (𝟑+𝟏)/(𝟗+𝟕) • 𝟐/𝟓 = 𝟎.𝟎𝟎𝟎𝟕𝟖
𝑷(𝒂𝒄𝒕𝒊𝒐𝒏|𝒇𝒂𝒔𝒕, 𝒇𝒖𝒓𝒊𝒐𝒖𝒔, 𝒇𝒖𝒏) = (𝟐+𝟏)/(𝟏𝟏+𝟕) • (𝟐+𝟏)/(𝟏𝟏+𝟕) • (𝟏+𝟏)/(𝟏𝟏+𝟕) • 𝟑/𝟓 = 𝟎.𝟎𝟎𝟏𝟖


EXAMPLE 2 : Playing tennis..
|---------------------------------------------------------------|
|Num    |Outlook    |Temperature    |Humidity   |Wind   |Class  |
|-------|-----------|---------------|-----------|-------|-------|
|1      |Sunny      |Hot            |High       |Weak   |No     |
|2      |Sunny      |Hot            |High       |Strong |No     |
|3      |Overcast   |Hot            |High       |Weak   |Yes    |
|4      |Rain       |Mild           |High       |Weak   |Yes    |
|5      |Rain       |Cool           |Normal     |Weak   |Yes    |
|6      |Rain       |Cool           |Normal     |Strong |No     |
|7      |Overcast   |Cool           |Normal     |Strong |Yes    |
|8      |Sunny      |Mild           |High       |Weak   |No     |
|9      |Sunny      |Cool           |Normal     |Weak   |Yes    |
|10     |Rain       |Mild           |Normal     |Weak   |Yes    |
|11     |Sunny      |Mild           |Normal     |Strong |Yes    |
|12     |Overcast   |Mild           |High       |Strong |Yes    |
|13     |Overcast   |Hot            |Normal     |Weak   |Yes    |
|14     |Rain       |Mild           |High       |Strong |No     |
|15     |Sunny      |Cool           |High       |Strong |???    |
|---------------------------------------------------------------|
𝑷(𝒚𝒆𝒔)=𝟗/𝟏𝟒,  𝑷(𝒏𝒐)=𝟓/𝟏𝟒
𝑷(𝒘𝒊𝒏𝒅=𝒔𝒕𝒓𝒐𝒏𝒈|𝒚𝒆𝒔)=𝟑/𝟗,  𝑷(𝒘𝒊𝒏𝒅=𝒔𝒕𝒓𝒐𝒏𝒈|𝒏𝒐)=𝟑/𝟓
...
𝑷(𝒚)𝑷(𝒔𝒖𝒏│𝒚)𝑷(𝒄𝒐𝒐𝒍│𝒚)𝑷(𝒉𝒊𝒈𝒉│𝒚)𝑷(𝒔𝒕𝒓𝒐𝒏𝒈│𝒚) = 𝟎.𝟎𝟎𝟓
𝑷(𝒏)𝑷(𝒔𝒖𝒏│𝒏)𝑷(𝒄𝒐𝒐𝒍│𝒏)𝑷(𝒉𝒊𝒈𝒉│𝒏)𝑷(𝒔𝒕𝒓𝒐𝒏𝒈│𝒏) = 𝟎.𝟎𝟐𝟏


EXAMPLE 3 : Male or female..
|---------------------------------------|
|Num    |Height |Weight |foot   |Class  |
|-------|-------|-------|-------|-------|
|1      |6      |180    |12     |Male   |
|2      |5.92   |190    |11     |Male   |
|3      |5.58   |170    |12     |Male   |
|4      |5.92   |165    |10     |Male   |
|5      |5      |100    |6      |Female |
|6      |5.5    |150    |8      |Female |
|7      |5.42   |130    |7      |Female |
|8      |5.75   |150    |9      |Female |
|9      |6      |130    |8      |???    |
|---------------------------------------|
Log likelihood
𝑷(class)𝑷(hei│class)𝑷(wei│class)𝑷(foot│class) ~ 
log( 𝑷(class)𝑷(hei│class)𝑷(wei│class)𝑷(foot│class) ) =
log(𝑷(class)) + log(𝑷(hei│class)) + log(𝑷(wei│class)) + log(𝑷(foot│class))
