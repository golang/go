// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math_test

import (
	"fmt"
	. "math"
	"testing"
)

var vf = []float64{
	4.9790119248836735e+00,
	7.7388724745781045e+00,
	-2.7688005719200159e-01,
	-5.0106036182710749e+00,
	9.6362937071984173e+00,
	2.9263772392439646e+00,
	5.2290834314593066e+00,
	2.7279399104360102e+00,
	1.8253080916808550e+00,
	-8.6859247685756013e+00,
}
// The expected results below were computed by the high precision calculators
// at http://keisan.casio.com/.  More exact input values (array vf[], above)
// were obtained by printing them with "%.26f".  The answers were calculated
// to 26 digits (by using the "Digit number" drop-down control of each
// calculator).  Twenty-six digits were chosen so that the answers would be
// accurate even for a float128 type.
var acos = []float64{
	1.0496193546107222142571536e+00,
	6.8584012813664425171660692e-01,
	1.5984878714577160325521819e+00,
	2.0956199361475859327461799e+00,
	2.7053008467824138592616927e-01,
	1.2738121680361776018155625e+00,
	1.0205369421140629186287407e+00,
	1.2945003481781246062157835e+00,
	1.3872364345374451433846657e+00,
	2.6231510803970463967294145e+00,
}
var acosh = []float64{
	2.4743347004159012494457618e+00,
	2.8576385344292769649802701e+00,
	7.2796961502981066190593175e-01,
	2.4796794418831451156471977e+00,
	3.0552020742306061857212962e+00,
	2.044238592688586588942468e+00,
	2.5158701513104513595766636e+00,
	1.99050839282411638174299e+00,
	1.6988625798424034227205445e+00,
	2.9611454842470387925531875e+00,
}
var asin = []float64{
	5.2117697218417440497416805e-01,
	8.8495619865825236751471477e-01,
	-02.769154466281941332086016e-02,
	-5.2482360935268931351485822e-01,
	1.3002662421166552333051524e+00,
	2.9698415875871901741575922e-01,
	5.5025938468083370060258102e-01,
	2.7629597861677201301553823e-01,
	1.83559892257451475846656e-01,
	-1.0523547536021497774980928e+00,
}
var asinh = []float64{
	2.3083139124923523427628243e+00,
	2.743551594301593620039021e+00,
	-2.7345908534880091229413487e-01,
	-2.3145157644718338650499085e+00,
	2.9613652154015058521951083e+00,
	1.7949041616585821933067568e+00,
	2.3564032905983506405561554e+00,
	1.7287118790768438878045346e+00,
	1.3626658083714826013073193e+00,
	-2.8581483626513914445234004e+00,
}
var atan = []float64{
	1.372590262129621651920085e+00,
	1.442290609645298083020664e+00,
	-2.7011324359471758245192595e-01,
	-1.3738077684543379452781531e+00,
	1.4673921193587666049154681e+00,
	1.2415173565870168649117764e+00,
	1.3818396865615168979966498e+00,
	1.2194305844639670701091426e+00,
	1.0696031952318783760193244e+00,
	-1.4561721938838084990898679e+00,
}
var atanh = []float64{
	5.4651163712251938116878204e-01,
	1.0299474112843111224914709e+00,
	-2.7695084420740135145234906e-02,
	-5.5072096119207195480202529e-01,
	1.9943940993171843235906642e+00,
	3.01448604578089708203017e-01,
	5.8033427206942188834370595e-01,
	2.7987997499441511013958297e-01,
	1.8459947964298794318714228e-01,
	-1.3273186910532645867272502e+00,
}
var atan2 = []float64{
	1.1088291730037004444527075e+00,
	9.1218183188715804018797795e-01,
	1.5984772603216203736068915e+00,
	2.0352918654092086637227327e+00,
	8.0391819139044720267356014e-01,
	1.2861075249894661588866752e+00,
	1.0889904479131695712182587e+00,
	1.3044821793397925293797357e+00,
	1.3902530903455392306872261e+00,
	2.2859857424479142655411058e+00,
}
var cbrt = []float64{
	1.7075799841925094446722675e+00,
	1.9779982212970353936691498e+00,
	-6.5177429017779910853339447e-01,
	-1.7111838886544019873338113e+00,
	2.1279920909827937423960472e+00,
	1.4303536770460741452312367e+00,
	1.7357021059106154902341052e+00,
	1.3972633462554328350552916e+00,
	1.2221149580905388454977636e+00,
	-2.0556003730500069110343596e+00,
}
var ceil = []float64{
	5.0000000000000000e+00,
	8.0000000000000000e+00,
	0.0000000000000000e+00,
	-5.0000000000000000e+00,
	1.0000000000000000e+01,
	3.0000000000000000e+00,
	6.0000000000000000e+00,
	3.0000000000000000e+00,
	2.0000000000000000e+00,
	-8.0000000000000000e+00,
}
var copysign = []float64{
	-4.9790119248836735e+00,
	-7.7388724745781045e+00,
	-2.7688005719200159e-01,
	-5.0106036182710749e+00,
	-9.6362937071984173e+00,
	-2.9263772392439646e+00,
	-5.2290834314593066e+00,
	-2.7279399104360102e+00,
	-1.8253080916808550e+00,
	-8.6859247685756013e+00,
}
var cos = []float64{
	2.634752140995199110787593e-01,
	1.148551260848219865642039e-01,
	9.6191297325640768154550453e-01,
	2.938141150061714816890637e-01,
	-9.777138189897924126294461e-01,
	-9.7693041344303219127199518e-01,
	4.940088096948647263961162e-01,
	-9.1565869021018925545016502e-01,
	-2.517729313893103197176091e-01,
	-7.39241351595676573201918e-01,
}
var cosh = []float64{
	7.2668796942212842775517446e+01,
	1.1479413465659254502011135e+03,
	1.0385767908766418550935495e+00,
	7.5000957789658051428857788e+01,
	7.655246669605357888468613e+03,
	9.3567491758321272072888257e+00,
	9.331351599270605471131735e+01,
	7.6833430994624643209296404e+00,
	3.1829371625150718153881164e+00,
	2.9595059261916188501640911e+03,
}
var erf = []float64{
	5.1865354817738701906913566e-01,
	7.2623875834137295116929844e-01,
	-3.123458688281309990629839e-02,
	-5.2143121110253302920437013e-01,
	8.2704742671312902508629582e-01,
	3.2101767558376376743993945e-01,
	5.403990312223245516066252e-01,
	3.0034702916738588551174831e-01,
	2.0369924417882241241559589e-01,
	-7.8069386968009226729944677e-01,
}
var erfc = []float64{
	4.8134645182261298093086434e-01,
	2.7376124165862704883070156e-01,
	1.0312345868828130999062984e+00,
	1.5214312111025330292043701e+00,
	1.7295257328687097491370418e-01,
	6.7898232441623623256006055e-01,
	4.596009687776754483933748e-01,
	6.9965297083261411448825169e-01,
	7.9630075582117758758440411e-01,
	1.7806938696800922672994468e+00,
}
var exp = []float64{
	1.4533071302642137507696589e+02,
	2.2958822575694449002537581e+03,
	7.5814542574851666582042306e-01,
	6.6668778421791005061482264e-03,
	1.5310493273896033740861206e+04,
	1.8659907517999328638667732e+01,
	1.8662167355098714543942057e+02,
	1.5301332413189378961665788e+01,
	6.2047063430646876349125085e+00,
	1.6894712385826521111610438e-04,
}
var expm1 = []float64{
	5.105047796122957327384770212e-02,
	8.046199708567344080562675439e-02,
	-2.764970978891639815187418703e-03,
	-4.8871434888875355394330300273e-02,
	1.0115864277221467777117227494e-01,
	2.969616407795910726014621657e-02,
	5.368214487944892300914037972e-02,
	2.765488851131274068067445335e-02,
	1.842068661871398836913874273e-02,
	-8.3193870863553801814961137573e-02,
}
var exp2 = []float64{
	3.1537839463286288034313104e+01,
	2.1361549283756232296144849e+02,
	8.2537402562185562902577219e-01,
	3.1021158628740294833424229e-02,
	7.9581744110252191462569661e+02,
	7.6019905892596359262696423e+00,
	3.7506882048388096973183084e+01,
	6.6250893439173561733216375e+00,
	3.5438267900243941544605339e+00,
	2.4281533133513300984289196e-03,
}
var fdim = []float64{
	4.9790119248836735e+00,
	7.7388724745781045e+00,
	0.0000000000000000e+00,
	0.0000000000000000e+00,
	9.6362937071984173e+00,
	2.9263772392439646e+00,
	5.2290834314593066e+00,
	2.7279399104360102e+00,
	1.8253080916808550e+00,
	0.0000000000000000e+00,
}
var floor = []float64{
	4.0000000000000000e+00,
	7.0000000000000000e+00,
	-1.0000000000000000e+00,
	-6.0000000000000000e+00,
	9.0000000000000000e+00,
	2.0000000000000000e+00,
	5.0000000000000000e+00,
	2.0000000000000000e+00,
	1.0000000000000000e+00,
	-9.0000000000000000e+00,
}
var fmod = []float64{
	4.197615023265299782906368e-02,
	2.261127525421895434476482e+00,
	3.231794108794261433104108e-02,
	4.989396381728925078391512e+00,
	3.637062928015826201999516e-01,
	1.220868282268106064236690e+00,
	4.770916568540693347699744e+00,
	1.816180268691969246219742e+00,
	8.734595415957246977711748e-01,
	1.314075231424398637614104e+00,
}

type fi struct {
	f float64
	i int
}

var frexp = []fi{
	fi{6.2237649061045918750e-01, 3},
	fi{9.6735905932226306250e-01, 3},
	fi{-5.5376011438400318000e-01, -1},
	fi{-6.2632545228388436250e-01, 3},
	fi{6.02268356699901081250e-01, 4},
	fi{7.3159430981099115000e-01, 2},
	fi{6.5363542893241332500e-01, 3},
	fi{6.8198497760900255000e-01, 2},
	fi{9.1265404584042750000e-01, 1},
	fi{-5.4287029803597508250e-01, 4},
}
var lgamma = []fi{
	fi{3.146492141244545774319734e+00, 1},
	fi{8.003414490659126375852113e+00, 1},
	fi{1.517575735509779707488106e+00, -1},
	fi{-2.588480028182145853558748e-01, 1},
	fi{1.1989897050205555002007985e+01, 1},
	fi{6.262899811091257519386906e-01, 1},
	fi{3.5287924899091566764846037e+00, 1},
	fi{4.5725644770161182299423372e-01, 1},
	fi{-6.363667087767961257654854e-02, 1},
	fi{-1.077385130910300066425564e+01, -1},
}
var log = []float64{
	1.605231462693062999102599e+00,
	2.0462560018708770653153909e+00,
	-1.2841708730962657801275038e+00,
	1.6115563905281545116286206e+00,
	2.2655365644872016636317461e+00,
	1.0737652208918379856272735e+00,
	1.6542360106073546632707956e+00,
	1.0035467127723465801264487e+00,
	6.0174879014578057187016475e-01,
	2.161703872847352815363655e+00,
}
var log10 = []float64{
	6.9714316642508290997617083e-01,
	8.886776901739320576279124e-01,
	-5.5770832400658929815908236e-01,
	6.998900476822994346229723e-01,
	9.8391002850684232013281033e-01,
	4.6633031029295153334285302e-01,
	7.1842557117242328821552533e-01,
	4.3583479968917773161304553e-01,
	2.6133617905227038228626834e-01,
	9.3881606348649405716214241e-01,
}
var log1p = []float64{
	4.8590257759797794104158205e-02,
	7.4540265965225865330849141e-02,
	-2.7726407903942672823234024e-03,
	-5.1404917651627649094953380e-02,
	9.1998280672258624681335010e-02,
	2.8843762576593352865894824e-02,
	5.0969534581863707268992645e-02,
	2.6913947602193238458458594e-02,
	1.8088493239630770262045333e-02,
	-9.0865245631588989681559268e-02,
}
var log2 = []float64{
	2.3158594707062190618898251e+00,
	2.9521233862883917703341018e+00,
	-1.8526669502700329984917062e+00,
	2.3249844127278861543568029e+00,
	3.268478366538305087466309e+00,
	1.5491157592596970278166492e+00,
	2.3865580889631732407886495e+00,
	1.447811865817085365540347e+00,
	8.6813999540425116282815557e-01,
	3.118679457227342224364709e+00,
}
var modf = [][2]float64{
	[2]float64{4.0000000000000000e+00, 9.7901192488367350108546816e-01},
	[2]float64{7.0000000000000000e+00, 7.3887247457810456552351752e-01},
	[2]float64{0.0000000000000000e+00, -2.7688005719200159404635997e-01},
	[2]float64{-5.0000000000000000e+00, -1.060361827107492160848778e-02},
	[2]float64{9.0000000000000000e+00, 6.3629370719841737980004837e-01},
	[2]float64{2.0000000000000000e+00, 9.2637723924396464525443662e-01},
	[2]float64{5.0000000000000000e+00, 2.2908343145930665230025625e-01},
	[2]float64{2.0000000000000000e+00, 7.2793991043601025126008608e-01},
	[2]float64{1.0000000000000000e+00, 8.2530809168085506044576505e-01},
	[2]float64{-8.0000000000000000e+00, -6.8592476857560136238589621e-01},
}
var nextafter = []float64{
	4.97901192488367438926388786e+00,
	7.73887247457810545370193722e+00,
	-2.7688005719200153853520874e-01,
	-5.01060361827107403343006808e+00,
	9.63629370719841915615688777e+00,
	2.92637723924396508934364647e+00,
	5.22908343145930754047867595e+00,
	2.72793991043601069534929593e+00,
	1.82530809168085528249036997e+00,
	-8.68592476857559958602905681e+00,
}
var pow = []float64{
	9.5282232631648411840742957e+04,
	5.4811599352999901232411871e+07,
	5.2859121715894396531132279e-01,
	9.7587991957286474464259698e-06,
	4.328064329346044846740467e+09,
	8.4406761805034547437659092e+02,
	1.6946633276191194947742146e+05,
	5.3449040147551939075312879e+02,
	6.688182138451414936380374e+01,
	2.0609869004248742886827439e-09,
}
var sin = []float64{
	-9.6466616586009283766724726e-01,
	9.9338225271646545763467022e-01,
	-2.7335587039794393342449301e-01,
	9.5586257685042792878173752e-01,
	-2.099421066779969164496634e-01,
	2.135578780799860532750616e-01,
	-8.694568971167362743327708e-01,
	4.019566681155577786649878e-01,
	9.6778633541687993721617774e-01,
	-6.734405869050344734943028e-01,
}
var sinh = []float64{
	7.2661916084208532301448439e+01,
	1.1479409110035194500526446e+03,
	-2.8043136512812518927312641e-01,
	-7.499429091181587232835164e+01,
	7.6552466042906758523925934e+03,
	9.3031583421672014313789064e+00,
	9.330815755828109072810322e+01,
	7.6179893137269146407361477e+00,
	3.021769180549615819524392e+00,
	-2.95950575724449499189888e+03,
}
var sqrt = []float64{
	2.2313699659365484748756904e+00,
	2.7818829009464263511285458e+00,
	5.2619393496314796848143251e-01,
	2.2384377628763938724244104e+00,
	3.1042380236055381099288487e+00,
	1.7106657298385224403917771e+00,
	2.286718922705479046148059e+00,
	1.6516476350711159636222979e+00,
	1.3510396336454586262419247e+00,
	2.9471892997524949215723329e+00,
}
var tan = []float64{
	-3.661316565040227801781974e+00,
	8.64900232648597589369854e+00,
	-2.8417941955033612725238097e-01,
	3.253290185974728640827156e+00,
	2.147275640380293804770778e-01,
	-2.18600910711067004921551e-01,
	-1.760002817872367935518928e+00,
	-4.389808914752818126249079e-01,
	-3.843885560201130679995041e+00,
	9.10988793377685105753416e-01,
}
var tanh = []float64{
	9.9990531206936338549262119e-01,
	9.9999962057085294197613294e-01,
	-2.7001505097318677233756845e-01,
	-9.9991110943061718603541401e-01,
	9.9999999146798465745022007e-01,
	9.9427249436125236705001048e-01,
	9.9994257600983138572705076e-01,
	9.9149409509772875982054701e-01,
	9.4936501296239685514466577e-01,
	-9.9999994291374030946055701e-01,
}
var trunc = []float64{
	4.0000000000000000e+00,
	7.0000000000000000e+00,
	-0.0000000000000000e+00,
	-5.0000000000000000e+00,
	9.0000000000000000e+00,
	2.0000000000000000e+00,
	5.0000000000000000e+00,
	2.0000000000000000e+00,
	1.0000000000000000e+00,
	-8.0000000000000000e+00,
}

// arguments and expected results for special cases
var vfacoshSC = []float64{
	Inf(-1),
	0.5,
	NaN(),
}
var acoshSC = []float64{
	NaN(),
	NaN(),
	NaN(),
}

var vfasinSC = []float64{
	NaN(),
	-Pi,
	Pi,
}
var asinSC = []float64{
	NaN(),
	NaN(),
	NaN(),
}

var vfasinhSC = []float64{
	Inf(-1),
	Inf(1),
	NaN(),
}
var asinhSC = []float64{
	Inf(-1),
	Inf(1),
	NaN(),
}

var vfatanSC = []float64{
	NaN(),
}
var atanSC = []float64{
	NaN(),
}

var vfatanhSC = []float64{
	-Pi,
	-1,
	1,
	Pi,
	NaN(),
}
var atanhSC = []float64{
	NaN(),
	Inf(-1),
	Inf(1),
	NaN(),
	NaN(),
}
var vfatan2SC = [][2]float64{
	[2]float64{Inf(-1), Inf(-1)},
	[2]float64{Inf(-1), -Pi},
	[2]float64{Inf(-1), 0},
	[2]float64{Inf(-1), +Pi},
	[2]float64{Inf(-1), Inf(1)},
	[2]float64{Inf(-1), NaN()},
	[2]float64{-Pi, Inf(-1)},
	[2]float64{-Pi, 0},
	[2]float64{-Pi, Inf(1)},
	[2]float64{-Pi, NaN()},
	[2]float64{0, Inf(-1)},
	[2]float64{0, -Pi},
	[2]float64{0, 0},
	[2]float64{0, +Pi},
	[2]float64{0, Inf(1)},
	[2]float64{0, NaN()},
	[2]float64{+Pi, Inf(-1)},
	[2]float64{+Pi, 0},
	[2]float64{+Pi, Inf(1)},
	[2]float64{+Pi, NaN()},
	[2]float64{Inf(1), Inf(-1)},
	[2]float64{Inf(1), -Pi},
	[2]float64{Inf(1), 0},
	[2]float64{Inf(1), +Pi},
	[2]float64{Inf(1), Inf(1)},
	[2]float64{Inf(1), NaN()},
	[2]float64{NaN(), NaN()},
}
var atan2SC = []float64{
	-3 * Pi / 4,
	-Pi / 2,
	-Pi / 2,
	-Pi / 2,
	-Pi / 4,
	NaN(),
	-Pi,
	-Pi / 2,
	-0,
	NaN(),
	Pi,
	Pi,
	0,
	0,
	0,
	NaN(),
	Pi,
	Pi / 2,
	0,
	NaN(),
	3 * Pi / 4,
	Pi / 2,
	Pi / 2,
	Pi / 2,
	Pi / 4,
	NaN(),
	NaN(),
}

var vfcbrtSC = []float64{
	Inf(-1),
	Inf(1),
	NaN(),
}
var cbrtSC = []float64{
	Inf(-1),
	Inf(1),
	NaN(),
}

var vfceilSC = []float64{
	Inf(-1),
	Inf(1),
	NaN(),
}
var ceilSC = []float64{
	Inf(-1),
	Inf(1),
	NaN(),
}

var vfcopysignSC = []float64{
	Inf(-1),
	Inf(1),
	NaN(),
}
var copysignSC = []float64{
	Inf(-1),
	Inf(-1),
	NaN(),
}

var vferfSC = []float64{
	Inf(-1),
	Inf(1),
	NaN(),
}
var erfSC = []float64{
	-1,
	1,
	NaN(),
}
var erfcSC = []float64{
	2,
	0,
	NaN(),
}

var vfexpSC = []float64{
	Inf(-1),
	Inf(1),
	NaN(),
}
var expSC = []float64{
	0,
	Inf(1),
	NaN(),
}
var expm1SC = []float64{
	-1,
	Inf(1),
	NaN(),
}

var vffmodSC = [][2]float64{
	[2]float64{Inf(-1), Inf(-1)},
	[2]float64{Inf(-1), -Pi},
	[2]float64{Inf(-1), 0},
	[2]float64{Inf(-1), Pi},
	[2]float64{Inf(-1), Inf(1)},
	[2]float64{Inf(-1), NaN()},
	[2]float64{-Pi, Inf(-1)},
	[2]float64{-Pi, 0},
	[2]float64{-Pi, Inf(1)},
	[2]float64{-Pi, NaN()},
	[2]float64{0, Inf(-1)},
	[2]float64{0, 0},
	[2]float64{0, Inf(1)},
	[2]float64{0, NaN()},
	[2]float64{Pi, Inf(-1)},
	[2]float64{Pi, 0},
	[2]float64{Pi, Inf(1)},
	[2]float64{Pi, NaN()},
	[2]float64{Inf(1), Inf(-1)},
	[2]float64{Inf(1), -Pi},
	[2]float64{Inf(1), 0},
	[2]float64{Inf(1), Pi},
	[2]float64{Inf(1), Inf(1)},
	[2]float64{Inf(1), NaN()},
	[2]float64{NaN(), Inf(-1)},
	[2]float64{NaN(), -Pi},
	[2]float64{NaN(), 0},
	[2]float64{NaN(), Pi},
	[2]float64{NaN(), Inf(1)},
	[2]float64{NaN(), NaN()},
}
var fmodSC = []float64{
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	-Pi,
	NaN(),
	-Pi,
	NaN(),
	0,
	NaN(),
	0,
	NaN(),
	Pi,
	NaN(),
	Pi,
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
}

var vffrexpSC = []float64{
	Inf(-1),
	0,
	Inf(1),
	NaN(),
}
var frexpSC = []fi{
	fi{Inf(-1), 0},
	fi{0, 0},
	fi{Inf(1), 0},
	fi{NaN(), 0},
}

var vfhypotSC = [][2]float64{
	[2]float64{Inf(-1), Inf(-1)},
	[2]float64{Inf(-1), 0},
	[2]float64{Inf(-1), Inf(1)},
	[2]float64{Inf(-1), NaN()},
	[2]float64{0, Inf(-1)},
	[2]float64{0, Inf(1)},
	[2]float64{0, NaN()},
	[2]float64{Inf(1), Inf(-1)},
	[2]float64{Inf(1), 0},
	[2]float64{Inf(1), Inf(1)},
	[2]float64{Inf(1), NaN()},
	[2]float64{NaN(), Inf(-1)},
	[2]float64{NaN(), 0},
	[2]float64{NaN(), Inf(1)},
	[2]float64{NaN(), NaN()},
}
var hypotSC = []float64{
	Inf(1),
	Inf(1),
	Inf(1),
	Inf(1),
	Inf(1),
	Inf(1),
	NaN(),
	Inf(1),
	Inf(1),
	Inf(1),
	Inf(1),
	Inf(1),
	NaN(),
	Inf(1),
	NaN(),
}

var vflgammaSC = []float64{
	Inf(-1),
	-3,
	0,
	Inf(1),
	NaN(),
}
var lgammaSC = []fi{
	fi{Inf(-1), 1},
	fi{Inf(1), 1},
	fi{Inf(1), 1},
	fi{Inf(1), 1},
	fi{NaN(), 1},
}

var vflogSC = []float64{
	Inf(-1),
	-Pi,
	0,
	Inf(1),
	NaN(),
}
var logSC = []float64{
	NaN(),
	NaN(),
	Inf(-1),
	Inf(1),
	NaN(),
}

var vflog1pSC = []float64{
	Inf(-1),
	-Pi,
	-1,
	Inf(1),
	NaN(),
}
var log1pSC = []float64{
	NaN(),
	NaN(),
	Inf(-1),
	Inf(1),
	NaN(),
}

var vfmodfSC = []float64{
	Inf(-1),
	Inf(1),
	NaN(),
}
var modfSC = [][2]float64{
	[2]float64{Inf(-1), NaN()},
	[2]float64{Inf(1), NaN()},
	[2]float64{NaN(), NaN()},
}

var vfnextafterSC = [][2]float64{
	[2]float64{0, NaN()},
	[2]float64{NaN(), 0},
	[2]float64{NaN(), NaN()},
}
var nextafterSC = []float64{
	NaN(),
	NaN(),
	NaN(),
}

var vfpowSC = [][2]float64{
	[2]float64{-Pi, Pi},
	[2]float64{-Pi, -Pi},
	[2]float64{Inf(-1), 3},
	[2]float64{Inf(-1), Pi},
	[2]float64{Inf(-1), -3},
	[2]float64{Inf(-1), -Pi},
	[2]float64{Inf(1), Pi},
	[2]float64{0, -Pi},
	[2]float64{Inf(1), -Pi},
	[2]float64{0, Pi},
	[2]float64{-1, Inf(-1)},
	[2]float64{-1, Inf(1)},
	[2]float64{1, Inf(-1)},
	[2]float64{1, Inf(1)},
	[2]float64{-1 / 2, Inf(1)},
	[2]float64{1 / 2, Inf(1)},
	[2]float64{-Pi, Inf(-1)},
	[2]float64{Pi, Inf(-1)},
	[2]float64{-1 / 2, Inf(-1)},
	[2]float64{1 / 2, Inf(-1)},
	[2]float64{-Pi, Inf(1)},
	[2]float64{Pi, Inf(1)},
	[2]float64{NaN(), -Pi},
	[2]float64{NaN(), Pi},
	[2]float64{Inf(-1), NaN()},
	[2]float64{-Pi, NaN()},
	[2]float64{0, NaN()},
	[2]float64{Pi, NaN()},
	[2]float64{Inf(1), NaN()},
	[2]float64{NaN(), NaN()},
	[2]float64{Inf(-1), 1},
	[2]float64{-Pi, 1},
	[2]float64{0, 1},
	[2]float64{Pi, 1},
	[2]float64{Inf(1), 1},
	[2]float64{NaN(), 1},
	[2]float64{Inf(-1), 0},
	[2]float64{-Pi, 0},
	[2]float64{0, 0},
	[2]float64{Pi, 0},
	[2]float64{Inf(1), 0},
	[2]float64{NaN(), 0},
}
var powSC = []float64{
	NaN(),
	NaN(),
	Inf(-1),
	Inf(1),
	0,
	0,
	Inf(1),
	Inf(1),
	0,
	0,
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	0,
	0,
	0,
	0,
	Inf(1),
	Inf(1),
	Inf(1),
	Inf(1),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	Inf(-1),
	-Pi,
	0,
	Pi,
	Inf(1),
	NaN(),
	1,
	1,
	1,
	1,
	1,
	1,
}

var vfsqrtSC = []float64{
	Inf(-1),
	-Pi,
	Inf(1),
	NaN(),
}
var sqrtSC = []float64{
	NaN(),
	NaN(),
	Inf(1),
	NaN(),
}

func tolerance(a, b, e float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}

	if a != 0 {
		e = e * a
		if e < 0 {
			e = -e
		}
	}
	return d < e
}
func kindaclose(a, b float64) bool { return tolerance(a, b, 1e-8) }
func close(a, b float64) bool      { return tolerance(a, b, 1e-14) }
func veryclose(a, b float64) bool  { return tolerance(a, b, 4e-16) }
func alike(a, b float64) bool {
	switch {
	case IsNaN(a) && IsNaN(b):
		return true
	case IsInf(a, 1) && IsInf(b, 1):
		return true
	case IsInf(a, -1) && IsInf(b, -1):
		return true
	case a == b:
		return true
	}
	return false
}

func TestAcos(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 10
		if f := Acos(a); !close(acos[i], f) {
			t.Errorf("Acos(%g) = %g, want %g\n", a, f, acos[i])
		}
	}
	for i := 0; i < len(vfasinSC); i++ {
		if f := Acos(vfasinSC[i]); !alike(asinSC[i], f) {
			t.Errorf("Acos(%g) = %g, want %g\n", vfasinSC[i], f, asinSC[i])
		}
	}
}

func TestAcosh(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := 1 + Fabs(vf[i])
		if f := Acosh(a); !veryclose(acosh[i], f) {
			t.Errorf("Acosh(%g) = %g, want %g\n", a, f, acosh[i])
		}
	}
	for i := 0; i < len(vfacoshSC); i++ {
		if f := Acosh(vfacoshSC[i]); !alike(acoshSC[i], f) {
			t.Errorf("Acosh(%g) = %g, want %g\n", vfacoshSC[i], f, acoshSC[i])
		}
	}
}

func TestAsin(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 10
		if f := Asin(a); !veryclose(asin[i], f) {
			t.Errorf("Asin(%g) = %g, want %g\n", a, f, asin[i])
		}
	}
	for i := 0; i < len(vfasinSC); i++ {
		if f := Asin(vfasinSC[i]); !alike(asinSC[i], f) {
			t.Errorf("Asin(%g) = %g, want %g\n", vfasinSC[i], f, asinSC[i])
		}
	}
}

func TestAsinh(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Asinh(vf[i]); !veryclose(asinh[i], f) {
			t.Errorf("Asinh(%g) = %g, want %g\n", vf[i], f, asinh[i])
		}
	}
	for i := 0; i < len(vfasinhSC); i++ {
		if f := Asinh(vfasinhSC[i]); !alike(asinhSC[i], f) {
			t.Errorf("Asinh(%g) = %g, want %g\n", vfasinhSC[i], f, asinhSC[i])
		}
	}
}

func TestAtan(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Atan(vf[i]); !veryclose(atan[i], f) {
			t.Errorf("Atan(%g) = %g, want %g\n", vf[i], f, atan[i])
		}
	}
	for i := 0; i < len(vfatanSC); i++ {
		if f := Atan(vfatanSC[i]); !alike(atanSC[i], f) {
			t.Errorf("Atan(%g) = %g, want %g\n", vfatanSC[i], f, atanSC[i])
		}
	}
}

func TestAtanh(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 10
		if f := Atanh(a); !veryclose(atanh[i], f) {
			t.Errorf("Atanh(%g) = %g, want %g\n", a, f, atanh[i])
		}
	}
	for i := 0; i < len(vfatanhSC); i++ {
		if f := Atanh(vfatanhSC[i]); !alike(atanhSC[i], f) {
			t.Errorf("Atanh(%g) = %g, want %g\n", vfatanhSC[i], f, atanhSC[i])
		}
	}
}

func TestAtan2(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Atan2(10, vf[i]); !veryclose(atan2[i], f) {
			t.Errorf("Atan2(10, %g) = %g, want %g\n", vf[i], f, atan2[i])
		}
	}
	for i := 0; i < len(vfatan2SC); i++ {
		if f := Atan2(vfatan2SC[i][0], vfatan2SC[i][1]); !alike(atan2SC[i], f) {
			t.Errorf("Atan2(%g, %g) = %g, want %g\n", vfatan2SC[i][0], vfatan2SC[i][1], f, atan2SC[i])
		}
	}
}

func TestCbrt(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Cbrt(vf[i]); !veryclose(cbrt[i], f) {
			t.Errorf("Cbrt(%g) = %g, want %g\n", vf[i], f, cbrt[i])
		}
	}
	for i := 0; i < len(vfcbrtSC); i++ {
		if f := Cbrt(vfcbrtSC[i]); !alike(cbrtSC[i], f) {
			t.Errorf("Cbrt(%g) = %g, want %g\n", vfcbrtSC[i], f, cbrtSC[i])
		}
	}
}

func TestCeil(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Ceil(vf[i]); ceil[i] != f {
			t.Errorf("Ceil(%g) = %g, want %g\n", vf[i], f, ceil[i])
		}
	}
	for i := 0; i < len(vfceilSC); i++ {
		if f := Ceil(vfceilSC[i]); !alike(ceilSC[i], f) {
			t.Errorf("Ceil(%g) = %g, want %g\n", vfceilSC[i], f, ceilSC[i])
		}
	}
}

func TestCopysign(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Copysign(vf[i], -1); copysign[i] != f {
			t.Errorf("Copysign(%g, -1) = %g, want %g\n", vf[i], f, copysign[i])
		}
	}
	for i := 0; i < len(vfcopysignSC); i++ {
		if f := Copysign(vfcopysignSC[i], -1); !alike(copysignSC[i], f) {
			t.Errorf("Copysign(%g, -1) = %g, want %g\n", vfcopysignSC[i], f, copysignSC[i])
		}
	}
}

func TestCos(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Cos(vf[i]); !close(cos[i], f) {
			t.Errorf("Cos(%g) = %g, want %g\n", vf[i], f, cos[i])
		}
	}
}

func TestCosh(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Cosh(vf[i]); !close(cosh[i], f) {
			t.Errorf("Cosh(%g) = %g, want %g\n", vf[i], f, cosh[i])
		}
	}
}

func TestErf(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 10
		if f := Erf(a); !veryclose(erf[i], f) {
			t.Errorf("Erf(%g) = %g, want %g\n", a, f, erf[i])
		}
	}
	for i := 0; i < len(vferfSC); i++ {
		if f := Erf(vferfSC[i]); !alike(erfSC[i], f) {
			t.Errorf("Erf(%g) = %g, want %g\n", vferfSC[i], f, erfSC[i])
		}
	}
}

func TestErfc(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 10
		if f := Erfc(a); !veryclose(erfc[i], f) {
			t.Errorf("Erfc(%g) = %g, want %g\n", a, f, erfc[i])
		}
	}
	for i := 0; i < len(vferfSC); i++ {
		if f := Erfc(vferfSC[i]); !alike(erfcSC[i], f) {
			t.Errorf("Erfc(%g) = %g, want %g\n", vferfSC[i], f, erfcSC[i])
		}
	}
}

func TestExp(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Exp(vf[i]); !close(exp[i], f) {
			t.Errorf("Exp(%g) = %g, want %g\n", vf[i], f, exp[i])
		}
	}
	for i := 0; i < len(vfexpSC); i++ {
		if f := Exp(vfexpSC[i]); !alike(expSC[i], f) {
			t.Errorf("Exp(%g) = %g, want %g\n", vfexpSC[i], f, expSC[i])
		}
	}
}

func TestExpm1(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 100
		if f := Expm1(a); !veryclose(expm1[i], f) {
			t.Errorf("Expm1(%.26fg) = %.26fg, want %.26fg\n", a, f, expm1[i])
		}
	}
	for i := 0; i < len(vfexpSC); i++ {
		if f := Expm1(vfexpSC[i]); !alike(expm1SC[i], f) {
			t.Errorf("Expm1(%g) = %g, want %g\n", vfexpSC[i], f, expm1SC[i])
		}
	}
}

func TestExp2(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Exp2(vf[i]); !close(exp2[i], f) {
			t.Errorf("Exp2(%g) = %g, want %g\n", vf[i], f, exp2[i])
		}
	}
	for i := 0; i < len(vfexpSC); i++ {
		if f := Exp2(vfexpSC[i]); !alike(expSC[i], f) {
			t.Errorf("Exp2(%g) = %g, want %g\n", vfexpSC[i], f, expSC[i])
		}
	}
}

func TestFdim(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Fdim(vf[i], 0); fdim[i] != f {
			t.Errorf("Fdim(%g, %g) = %g, want %g\n", vf[i], 0.0, f, fdim[i])
		}
	}
}

func TestFloor(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Floor(vf[i]); floor[i] != f {
			t.Errorf("Floor(%g) = %g, want %g\n", vf[i], f, floor[i])
		}
	}
	for i := 0; i < len(vfceilSC); i++ {
		if f := Floor(vfceilSC[i]); !alike(ceilSC[i], f) {
			t.Errorf("Floor(%g) = %g, want %g\n", vfceilSC[i], f, ceilSC[i])
		}
	}
}

func TestFmax(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Fmax(vf[i], ceil[i]); ceil[i] != f {
			t.Errorf("Fmax(%g, %g) = %g, want %g\n", vf[i], ceil[i], f, ceil[i])
		}
	}
}

func TestFmin(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Fmin(vf[i], floor[i]); floor[i] != f {
			t.Errorf("Fmin(%g, %g) = %g, want %g\n", vf[i], floor[i], f, floor[i])
		}
	}
}

func TestFmod(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Fmod(10, vf[i]); fmod[i] != f { /*!close(fmod[i], f)*/
							t.Errorf("Fmod(10, %g) = %g, want %g\n", vf[i], f, fmod[i])
		}
	}
	for i := 0; i < len(vffmodSC); i++ {
		if f := Fmod(vffmodSC[i][0], vffmodSC[i][1]); !alike(fmodSC[i], f) {
			t.Errorf("Fmod(%g, %g) = %g, want %g\n", vffmodSC[i][0], vffmodSC[i][1], f, fmodSC[i])
		}
	}
}

func TestFrexp(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f, j := Frexp(vf[i]); !veryclose(frexp[i].f, f) || frexp[i].i != j {
			t.Errorf("Frexp(%g) = %g, %d, want %g, %d\n", vf[i], f, j, frexp[i].f, frexp[i].i)
		}
	}
	for i := 0; i < len(vffrexpSC); i++ {
		if f, j := Frexp(vffrexpSC[i]); !alike(frexpSC[i].f, f) || frexpSC[i].i != j {
			t.Errorf("Frexp(%g) = %g, %d, want %g, %d\n", vffrexpSC[i], f, j, frexpSC[i].f, frexpSC[i].i)
		}
	}
}

func TestHypot(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := Fabs(1e200 * tanh[i] * Sqrt(2))
		if f := Hypot(1e200*tanh[i], 1e200*tanh[i]); !veryclose(a, f) {
			t.Errorf("Hypot(%g, %g) = %g, want %g\n", 1e200*tanh[i], 1e200*tanh[i], f, a)
		}
	}
	for i := 0; i < len(vfhypotSC); i++ {
		if f := Hypot(vfhypotSC[i][0], vfhypotSC[i][1]); !alike(hypotSC[i], f) {
			t.Errorf("Hypot(%g, %g) = %g, want %g\n", vfhypotSC[i][0], vfhypotSC[i][1], f, hypotSC[i])
		}
	}
}

func TestLdexp(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Ldexp(frexp[i].f, frexp[i].i); !veryclose(vf[i], f) {
			t.Errorf("Ldexp(%g, %d) = %g, want %g\n", frexp[i].f, frexp[i].i, f, vf[i])
		}
	}
	for i := 0; i < len(vffrexpSC); i++ {
		if f := Ldexp(frexpSC[i].f, frexpSC[i].i); !alike(vffrexpSC[i], f) {
			t.Errorf("Ldexp(%g, %d) = %g, want %g\n", frexpSC[i].f, frexpSC[i].i, f, vffrexpSC[i])
		}
	}
}

func TestLgamma(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f, s := Lgamma(vf[i]); !close(lgamma[i].f, f) || lgamma[i].i != s {
			t.Errorf("Lgamma(%g) = %g, %d, want %g, %d\n", vf[i], f, s, lgamma[i].f, lgamma[i].i)
		}
	}
	for i := 0; i < len(vflgammaSC); i++ {
		if f, s := Lgamma(vflgammaSC[i]); !alike(lgammaSC[i].f, f) || lgammaSC[i].i != s {
			t.Errorf("Lgamma(%g) = %g, %d, want %g, %d\n", vflgammaSC[i], f, s, lgammaSC[i].f, lgammaSC[i].i)
		}
	}
}

func TestLog(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := Fabs(vf[i])
		if f := Log(a); log[i] != f {
			t.Errorf("Log(%g) = %g, want %g\n", a, f, log[i])
		}
	}
	if f := Log(10); f != Ln10 {
		t.Errorf("Log(%g) = %g, want %g\n", 10.0, f, Ln10)
	}
	for i := 0; i < len(vflogSC); i++ {
		if f := Log(vflogSC[i]); !alike(logSC[i], f) {
			t.Errorf("Log(%g) = %g, want %g\n", vflogSC[i], f, logSC[i])
		}
	}
}

func TestLog10(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := Fabs(vf[i])
		if f := Log10(a); !veryclose(log10[i], f) {
			t.Errorf("Log10(%g) = %g, want %g\n", a, f, log10[i])
		}
	}
	if f := Log10(E); f != Log10E {
		t.Errorf("Log10(%g) = %g, want %g\n", E, f, Log10E)
	}
	for i := 0; i < len(vflogSC); i++ {
		if f := Log10(vflogSC[i]); !alike(logSC[i], f) {
			t.Errorf("Log10(%g) = %g, want %g\n", vflogSC[i], f, logSC[i])
		}
	}
}

func TestLog1p(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 100
		if f := Log1p(a); !veryclose(log1p[i], f) {
			t.Errorf("Log1p(%g) = %g, want %g\n", a, f, log1p[i])
		}
	}
	a := float64(9)
	if f := Log1p(a); f != Ln10 {
		t.Errorf("Log1p(%g) = %g, want %g\n", a, f, Ln10)
	}
	for i := 0; i < len(vflogSC); i++ {
		if f := Log1p(vflog1pSC[i]); !alike(log1pSC[i], f) {
			t.Errorf("Log1p(%g) = %g, want %g\n", vflog1pSC[i], f, log1pSC[i])
		}
	}
}

func TestLog2(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := Fabs(vf[i])
		if f := Log2(a); !veryclose(log2[i], f) {
			t.Errorf("Log2(%g) = %g, want %g\n", a, f, log2[i])
		}
	}
	if f := Log2(E); f != Log2E {
		t.Errorf("Log2(%g) = %g, want %g\n", E, f, Log2E)
	}
	for i := 0; i < len(vflogSC); i++ {
		if f := Log2(vflogSC[i]); !alike(logSC[i], f) {
			t.Errorf("Log2(%g) = %g, want %g\n", vflogSC[i], f, logSC[i])
		}
	}
}

func TestModf(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f, g := Modf(vf[i]); !veryclose(modf[i][0], f) || !veryclose(modf[i][1], g) {
			t.Errorf("Modf(%g) = %g, %g, want %g, %g\n", vf[i], f, g, modf[i][0], modf[i][1])
		}
	}
	for i := 0; i < len(vfmodfSC); i++ {
		if f, g := Modf(vfmodfSC[i]); !alike(modfSC[i][0], f) || !alike(modfSC[i][1], g) {
			t.Errorf("Modf(%g) = %g, %g, want %g, %g\n", vfmodfSC[i], f, g, modfSC[i][0], modfSC[i][1])
		}
	}
}

func TestNextafter(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Nextafter(vf[i], 10); nextafter[i] != f {
			t.Errorf("Nextafter(%g, %g) = %g want %g\n", vf[i], 10.0, f, nextafter[i])
		}
	}
	for i := 0; i < len(vfmodfSC); i++ {
		if f := Nextafter(vfnextafterSC[i][0], vfnextafterSC[i][1]); !alike(nextafterSC[i], f) {
			t.Errorf("Nextafter(%g, %g) = %g want %g\n", vfnextafterSC[i][0], vfnextafterSC[i][1], f, nextafterSC[i])
		}
	}
}

func TestPow(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Pow(10, vf[i]); !close(pow[i], f) {
			t.Errorf("Pow(10, %g) = %g, want %g\n", vf[i], f, pow[i])
		}
	}
	for i := 0; i < len(vfpowSC); i++ {
		if f := Pow(vfpowSC[i][0], vfpowSC[i][1]); !alike(powSC[i], f) {
			t.Errorf("Pow(%g, %g) = %g, want %g\n", vfpowSC[i][0], vfpowSC[i][1], f, powSC[i])
		}
	}
}

func TestSin(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Sin(vf[i]); !close(sin[i], f) {
			t.Errorf("Sin(%g) = %g, want %g\n", vf[i], f, sin[i])
		}
	}
}

func TestSincos(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if s, c := Sincos(vf[i]); !close(sin[i], s) || !close(cos[i], c) {
			t.Errorf("Sincos(%g) = %g, %g want %g, %g\n", vf[i], s, c, sin[i], cos[i])
		}
	}
}

func TestSinh(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Sinh(vf[i]); !close(sinh[i], f) {
			t.Errorf("Sinh(%g) = %g, want %g\n", vf[i], f, sinh[i])
		}
	}
}

func TestSqrt(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := Fabs(vf[i])
		if f := SqrtGo(a); sqrt[i] != f {
			t.Errorf("sqrtGo(%g) = %g, want %g\n", a, f, sqrt[i])
		}
		a = Fabs(vf[i])
		if f := Sqrt(a); sqrt[i] != f {
			t.Errorf("Sqrt(%g) = %g, want %g\n", a, f, sqrt[i])
		}
	}
	for i := 0; i < len(vfsqrtSC); i++ {
		if f := Log10(vfsqrtSC[i]); !alike(sqrtSC[i], f) {
			t.Errorf("Sqrt(%g) = %g, want %g\n", vfsqrtSC[i], f, sqrtSC[i])
		}
	}
}

func TestTan(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Tan(vf[i]); !close(tan[i], f) {
			t.Errorf("Tan(%g) = %g, want %g\n", vf[i], f, tan[i])
		}
	}
}

func TestTanh(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Tanh(vf[i]); !veryclose(tanh[i], f) {
			t.Errorf("Tanh(%g) = %g, want %g\n", vf[i], f, tanh[i])
		}
	}
}

func TestTrunc(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Trunc(vf[i]); trunc[i] != f {
			t.Errorf("Trunc(%g) = %g, want %g\n", vf[i], f, trunc[i])
		}
	}
	for i := 0; i < len(vfceilSC); i++ {
		if f := Trunc(vfceilSC[i]); !alike(ceilSC[i], f) {
			t.Errorf("Trunc(%g) = %g, want %g\n", vfceilSC[i], f, ceilSC[i])
		}
	}
}

// Check that math functions of high angle values
// return similar results to low angle values
func TestLargeCos(t *testing.T) {
	large := float64(100000 * Pi)
	for i := 0; i < len(vf); i++ {
		f1 := Cos(vf[i])
		f2 := Cos(vf[i] + large)
		if !kindaclose(f1, f2) {
			t.Errorf("Cos(%g) = %g, want %g\n", vf[i]+large, f2, f1)
		}
	}
}

func TestLargeSin(t *testing.T) {
	large := float64(100000 * Pi)
	for i := 0; i < len(vf); i++ {
		f1 := Sin(vf[i])
		f2 := Sin(vf[i] + large)
		if !kindaclose(f1, f2) {
			t.Errorf("Sin(%g) = %g, want %g\n", vf[i]+large, f2, f1)
		}
	}
}

func TestLargeSincos(t *testing.T) {
	large := float64(100000 * Pi)
	for i := 0; i < len(vf); i++ {
		f1, g1 := Sincos(vf[i])
		f2, g2 := Sincos(vf[i] + large)
		if !kindaclose(f1, f2) || !kindaclose(g1, g2) {
			t.Errorf("Sincos(%g) = %g, %g, want %g, %g\n", vf[i]+large, f2, g2, f1, g1)
		}
	}
}

func TestLargeTan(t *testing.T) {
	large := float64(100000 * Pi)
	for i := 0; i < len(vf); i++ {
		f1 := Tan(vf[i])
		f2 := Tan(vf[i] + large)
		if !kindaclose(f1, f2) {
			t.Errorf("Tan(%g) = %g, want %g\n", vf[i]+large, f2, f1)
		}
	}
}

// Check that math constants are accepted by compiler
// and have right value (assumes strconv.Atof works).
// http://code.google.com/p/go/issues/detail?id=201

type floatTest struct {
	val  interface{}
	name string
	str  string
}

var floatTests = []floatTest{
	floatTest{float64(MaxFloat64), "MaxFloat64", "1.7976931348623157e+308"},
	floatTest{float64(MinFloat64), "MinFloat64", "5e-324"},
	floatTest{float32(MaxFloat32), "MaxFloat32", "3.4028235e+38"},
	floatTest{float32(MinFloat32), "MinFloat32", "1e-45"},
}

func TestFloatMinMax(t *testing.T) {
	for _, tt := range floatTests {
		s := fmt.Sprint(tt.val)
		if s != tt.str {
			t.Errorf("Sprint(%v) = %s, want %s", tt.name, s, tt.str)
		}
	}
}

// Benchmarks

func BenchmarkAcos(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Acos(.5)
	}
}

func BenchmarkAcosh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Acosh(1.5)
	}
}

func BenchmarkAsin(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Asin(.5)
	}
}

func BenchmarkAsinh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Asinh(.5)
	}
}

func BenchmarkAtan(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Atan(.5)
	}
}

func BenchmarkAtanh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Atanh(.5)
	}
}

func BenchmarkAtan2(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Atan2(.5, 1)
	}
}

func BenchmarkCbrt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Cbrt(10)
	}
}

func BenchmarkCeil(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Ceil(.5)
	}
}

func BenchmarkCopysign(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Copysign(.5, -1)
	}
}

func BenchmarkCos(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Cos(.5)
	}
}

func BenchmarkCosh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Cosh(2.5)
	}
}

func BenchmarkErf(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Erf(.5)
	}
}

func BenchmarkErfc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Erfc(.5)
	}
}

func BenchmarkExp(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Exp(.5)
	}
}

func BenchmarkExpm1(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Expm1(.5)
	}
}

func BenchmarkExp2(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Exp(.5)
	}
}

func BenchmarkFloor(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Floor(.5)
	}
}

func BenchmarkFdim(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Fdim(10, 3)
	}
}

func BenchmarkFmax(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Fmax(10, 3)
	}
}

func BenchmarkFmin(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Fmin(10, 3)
	}
}

func BenchmarkFmod(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Fmod(10, 3)
	}
}

func BenchmarkFrexp(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Frexp(8)
	}
}

func BenchmarkHypot(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Hypot(3, 4)
	}
}

func BenchmarkLdexp(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Ldexp(.5, 2)
	}
}

func BenchmarkLgamma(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Lgamma(2.5)
	}
}

func BenchmarkLog(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Log(.5)
	}
}

func BenchmarkLog10(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Log10(.5)
	}
}

func BenchmarkLog1p(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Log1p(.5)
	}
}

func BenchmarkLog2(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Log2(.5)
	}
}

func BenchmarkModf(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Modf(1.5)
	}
}

func BenchmarkNextafter(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Nextafter(.5, 1)
	}
}

func BenchmarkPowInt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Pow(2, 2)
	}
}

func BenchmarkPowFrac(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Pow(2.5, 1.5)
	}
}

func BenchmarkSin(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sin(.5)
	}
}

func BenchmarkSincos(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sincos(.5)
	}
}

func BenchmarkSinh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sinh(2.5)
	}
}

func BenchmarkSqrt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sqrt(10)
	}
}

func BenchmarkSqrtGo(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SqrtGo(10)
	}
}

func BenchmarkTan(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Tan(.5)
	}
}

func BenchmarkTanh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Tanh(2.5)
	}
}
func BenchmarkTrunc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Trunc(.5)
	}
}
