#!/usr/bin/env python3

from operator import itemgetter

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import curve_fit

import itertools

line = itertools.cycle(['--', '-.', '-',])
marker = itertools.cycle((',', '+', '.', 'o','^','8')) 
fill = itertools.cycle(['top', 'none', 'full', 'left', 'right'])

populations={ # https://www.worldometers.info/world-population/population-by-country/
"China":1439323776,
"India":1380004385,
"United States":331002651,
"Indonesia":273523615,
"Pakistan":220892340,
"Brazil":212559417,
"Nigeria":206139589,
"Bangladesh":164689383,
"Russia":145934462,
"Mexico":128932753,
"Japan":126476461,
"Ethiopia":114963588,
"Philippines":109581078,
"Egypt":102334404,
"Vietnam":97338579,
"DR Congo":89561403,
"Turkey":84339067,
"Iran":83992949,
"Germany":83783942,
"Thailand":69799978,
"United Kingdom":67886011,
"France":65273511,
"Italy":60461826,
"Tanzania":59734218,
"South Africa":59308690,
"Myanmar":54409800,
"Kenya":53771296,
"South Korea":51269185,
"Colombia":50882891,
"Spain":46754778,
"Uganda":45741007,
"Argentina":45195774,
"Algeria":43851044,
"Sudan":43849260,
"Ukraine":43733762,
"Iraq":40222493,
"Afghanistan":38928346,
"Poland":37846611,
"Canada":37742154,
"Morocco":36910560,
"Saudi Arabia":34813871,
"Uzbekistan":33469203,
"Peru":32971854,
"Angola":32866272,
"Malaysia":32365999,
"Mozambique":31255435,
"Ghana":31072940,
"Yemen":29825964,
"Nepal":29136808,
"Venezuela":28435940,
"Madagascar":27691018,
"Cameroon":26545863,
"North Korea":25778816,
"Australia":25499884,
"Niger":24206644,
"Taiwan":23816775,
"Sri Lanka":21413249,
"Burkina Faso":20903273,
"Mali":20250833,
"Romania":19237691,
"Malawi":19129952,
"Chile":19116201,
"Kazakhstan":18776707,
"Zambia":18383955,
"Guatemala":17915568,
"Ecuador":17643054,
"Syria":17500658,
"Netherlands":17134872,
"Senegal":16743927,
"Cambodia":16718965,
"Chad":16425864,
"Somalia":15893222,
"Zimbabwe":14862924,
"Guinea":13132795,
"Rwanda":12952218,
"Benin":12123200,
"Burundi":11890784,
"Tunisia":11818619,
"Bolivia":11673021,
"Belgium":11589623,
"Haiti":11402528,
"Cuba":11326616,
"South Sudan":11193725,
"Dominican Republic":10847910,
"Czech Republic":10708981,
"Greece":10423054,
"Jordan":10203134,
"Portugal":10196709,
"Azerbaijan":10139177,
"Sweden":10099265,
"Honduras":9904607,
"United Arab Emirates":9890402,
"Hungary":9660351,
"Tajikistan":9537645,
"Belarus":9449323,
"Austria":9006398,
"Papua New Guinea":8947024,
"Serbia":8737371,
"Israel":8655535,
"Switzerland":8654622,
"Togo":8278724,
"Sierra Leone":7976983,
"Hong Kong":7496981,
"Laos":7275560,
"Paraguay":7132538,
"Bulgaria":6948445,
"Libya":6871292,
"Lebanon":6825445,
"Nicaragua":6624554,
"Kyrgyzstan":6524195,
"El Salvador":6486205,
"Turkmenistan":6031200,
"Singapore":5850342,
"Denmark":5792202,
"Finland":5540720,
"Congo":5518087,
"Slovakia":5459642,
"Norway":5421241,
"Oman":5106626,
"State of Palestine":5101414,
"Costa Rica":5094118,
"Liberia":5057681,
"Ireland":4937786,
"Central African Republic":4829767,
"New Zealand":4822233,
"Mauritania":4649658,
"Panama":4314767,
"Kuwait":4270571,
"Croatia":4105267,
"Moldova":4033963,
"Georgia":3989167,
"Eritrea":3546421,
"Uruguay":3473730,
"Bosnia and Herzegovina":3280819,
"Mongolia":3278290,
"Armenia":2963243,
"Jamaica":2961167,
"Qatar":2881053,
"Albania":2877797,
"Puerto Rico":2860853,
"Lithuania":2722289,
"Namibia":2540905,
"Gambia":2416668,
"Botswana":2351627,
"Gabon":2225734,
"Lesotho":2142249,
"North Macedonia":2083374,
"Slovenia":2078938,
"Guinea-Bissau":1968001,
"Latvia":1886198,
"Bahrain":1701575,
"Equatorial Guinea":1402985,
"Trinidad and Tobago":1399488,
"Estonia":1326535,
"Timor-Leste":1318445,
"Mauritius":1271768,
"Cyprus":1207359,
"Eswatini":1160164,
"Djibouti":988000,
"Fiji":896445,
"Réunion":895312,
"Comoros":869601,
"Guyana":786552,
"Bhutan":771608,
"Solomon Islands":686884,
"Macao":649335,
"Montenegro":628066,
"Luxembourg":625978,
"Western Sahara":597339,
"Suriname":586632,
"Cabo Verde":555987,
"Maldives":540544,
"Malta":441543,
"Brunei":437479,
"Guadeloupe":400124,
"Belize":397628,
"Bahamas":393244,
"Martinique":375265,
"Iceland":341243,
"Vanuatu":307145,
"French Guiana":298682,
"Barbados":287375,
"New Caledonia":285498,
"French Polynesia":280908,
"Mayotte":272815,
"Sao Tome & Principe":219159,
"Samoa":198414,
"Saint Lucia":183627,
"Channel Islands":173863,
"Guam":168775,
"Curaçao":164093,
"Kiribati":119449,
"Micronesia":115023,
"Grenada":112523,
"St. Vincent & Grenadines":110940,
"Aruba":106766,
"Tonga":105695,
"U.S. Virgin Islands":104425,
"Seychelles":98347,
"Antigua and Barbuda":97929,
"Isle of Man":85033,
"Andorra":77265,
"Dominica":71986,
"Cayman Islands":65722,
}

# datetime.datetime.strptime('Feb 17 2020', '%b %d %Y')
countries=[ # data from https://www.worldometers.info/coronavirus/
{
    "country": "Italy",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,1,1,1,4,4,1,5,4,8,12,11,27,28,41,49,36,133,97,168,196,189,250,175,368,349,345,475,427,627,793,651,601,743,683,712,919,889]
},
{
    "country": "United States",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,5,3,2,1,3,4,3,4,4,8,3,7,9,12,18,23,40,56,49,46,113,141,225,247,268,400,525]
},
{
    "country": "China",
    "dates": ["Jan 22","Jan 23","Jan 24","Jan 25","Jan 26","Jan 27","Jan 28","Jan 29","Jan 30","Jan 31","Feb 01","Feb 02","Feb 03","Feb 04","Feb 05","Feb 06","Feb 07","Feb 08","Feb 09","Feb 10","Feb 11","Feb 12","Feb 13","Feb 14","Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,8,16,15,24,26,26,38,43,46,45,57,64,65,73,73,86,89,97,108,97,146,121,143,142,105,98,136,114,118,109,97,150,71,52,29,44,47,35,42,31,38,31,30,28,27,22,17,22,11,7,13,10,14,13,11,8,3,7,6,9,7,4,6,5,3,5],
},
{
    "country": "Spain",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,5,2,7,13,6,19,31,47,63,98,48,191,105,193,262,288,391,539,680,656,718,773,844],
},
{
    "country": "Germany",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,3,2,1,4,4,9,2,16,24,16,10,29,36,47,61,84,82],
},
{
    "country": "France",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,3,2,7,3,11,3,15,13,18,12,36,21,27,89,108,78,112,112,186,240,231,365,299,319],
},
{
    "country": "Iran",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,2,2,2,4,4,3,7,8,9,11,12,11,15,16,16,21,49,43,54,63,75,85,97,113,129,135,147,149,149,123,129,127,122,143,157,144,139],
},
{
    "country": "United Kingdom",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,2,1,2,2,1,10,14,20,16,33,40,33,56,48,54,87,41,115,181,260],
},
{
    "country": "Switzerland",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,1,3,4,2,1,5,8,6,10,13,24,18,22,2,31,39,39,33]
},
{
    "country": "Netherlands",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,2,1,0,1,0,5,2,8,4,19,15,18,30,30,43,34,63,80,78,112,93],
},
{
    "country": "South Korea",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,1,1,0,4,2,3,1,1,3,1,4,7,4,3,7,1,5,2,3,7,0,6,1,5,3,0,6,3,7,3,8,2,7,9,6,5,8,5],
},
{
    "country": "Belgium",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,1,0,6,0,4,7,16,30,8,13,34,56,42,69,64],
},
{
    "country": "Turkey",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,5,12,9,7,7,15,16,17,16],
},
{
    "country": "Canada",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,3,4,1,3,0,7,1,4,2,10,3,16,5],
},
{
    "country": "Portugal",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,2,2,6,2,9,10,10,17,16,24],
},
{
    "country": "Norway",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,2,0,0,0,3,1,0,0,0,3,2,2,0,5,4],
},
{
    "country": "Brazil",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,3,4,7,7,9,12,13,18,15,22] ,
},
{
    "country": "Australia",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,2,0,0,1,1,0,0,0,0,1,3,2,0,1] ,
},
{
    "country": "Israel",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,2,2,3,4,0] ,
},
{
    "country": "Sweden",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,4,1,2,1,5,4,1,6,13,22,15,28,0],
},
{
    "country": "Denmark",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,0,0,2,3,4,0,11,8,2,7,11,13] ,
},
{
    "country": "Austria",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,2,1,0,2,0,2,8,5,7,3,18,9,10] ,
},
{
    "country": "Ireland",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,4,7,5,1,2,3,10,9,27,20,39,41,53,69,74,191,126,102,121,219,204,235,255,302,294] ,
},
{
    "country": "Norway",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,3,2,9,4,6,8,26,35,33,29,20,51,173,229,171,196,113,147,92,123,120,199,169,205,221,240,241,218,288,399,244]  ,
},
{
    "country": "Greece",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,2,1,0,1,0,1,4,3,2,2,3,2,5,1,4] ,
},
{
    "country": "Iraq",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,0,2,1,0,0,1,1,1,0,0,1,1,1,4,0,3,3,4,2,7,4,2] ,
},
{
    "country": "Malaysia",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,1,5,2,4,2,4,3,3,1],
},
{
    "country": "Algeria",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,1,0,1,2,2,2,4,2,0,2,2,4,1,3] ,
},
{
    "country": "Hong Kong",
    "dates": ["Feb 15","Feb 16","Feb 17","Feb 18","Feb 19","Feb 20","Feb 21","Feb 22","Feb 23","Feb 24","Feb 25","Feb 26","Feb 27","Feb 28","Feb 29","Mar 01","Mar 02","Mar 03","Mar 04","Mar 05","Mar 06","Mar 07","Mar 08","Mar 09","Mar 10","Mar 11","Mar 12","Mar 13","Mar 14","Mar 15","Mar 16","Mar 17","Mar 18","Mar 19","Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25","Mar 26","Mar 27","Mar 28"],
    "deaths": [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] ,
},
]


fig=plt.figure()
ax=fig.add_subplot(111)

for country in countries:

    first_day_index = 0
    for i in range(len(country["dates"])):
        if country["deaths"][i] != 0:
            first_day_index = i
            break
    days = [x for x in range(len(country["dates"]) - first_day_index+1)]
    deaths = [0]+country["deaths"][first_day_index:]
    pop = populations[country["country"]]
    deaths = [x/pop*1E6 for x in deaths]

    # print(country["country"])
    # print(len(days) == len(deaths))
    # print()
    # print(days)
    # print(deaths)

    ax.plot(days, deaths, 
        ls=next(line), 
        label=country["country"], 
        fillstyle=next(fill), 
        marker=next(marker))

ax.set_xlabel("Days Since First Reported Death")
ax.set_ylabel("Number of Deaths per 1 Million people (Non Cumulative)")
ax.set_title("Corona virus deaths by country")


plt.legend()
plt.show()
