// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmplx

import (
	"math"
	"testing"
)

// The higher-precision values in vc26 were used to derive the
// input arguments vc (see also comment below). For reference
// only (do not delete).
var vc26 = []complex128{
	(4.97901192488367350108546816 + 7.73887247457810456552351752i),
	(7.73887247457810456552351752 - 0.27688005719200159404635997i),
	(-0.27688005719200159404635997 - 5.01060361827107492160848778i),
	(-5.01060361827107492160848778 + 9.63629370719841737980004837i),
	(9.63629370719841737980004837 + 2.92637723924396464525443662i),
	(2.92637723924396464525443662 + 5.22908343145930665230025625i),
	(5.22908343145930665230025625 + 2.72793991043601025126008608i),
	(2.72793991043601025126008608 + 1.82530809168085506044576505i),
	(1.82530809168085506044576505 - 8.68592476857560136238589621i),
	(-8.68592476857560136238589621 + 4.97901192488367350108546816i),
}

var vc = []complex128{
	(4.9790119248836735e+00 + 7.7388724745781045e+00i),
	(7.7388724745781045e+00 - 2.7688005719200159e-01i),
	(-2.7688005719200159e-01 - 5.0106036182710749e+00i),
	(-5.0106036182710749e+00 + 9.6362937071984173e+00i),
	(9.6362937071984173e+00 + 2.9263772392439646e+00i),
	(2.9263772392439646e+00 + 5.2290834314593066e+00i),
	(5.2290834314593066e+00 + 2.7279399104360102e+00i),
	(2.7279399104360102e+00 + 1.8253080916808550e+00i),
	(1.8253080916808550e+00 - 8.6859247685756013e+00i),
	(-8.6859247685756013e+00 + 4.9790119248836735e+00i),
}

// The expected results below were computed by the high precision calculators
// at https://keisan.casio.com/.  More exact input values (array vc[], above)
// were obtained by printing them with "%.26f".  The answers were calculated
// to 26 digits (by using the "Digit number" drop-down control of each
// calculator).

var abs = []float64{
	9.2022120669932650313380972e+00,
	7.7438239742296106616261394e+00,
	5.0182478202557746902556648e+00,
	1.0861137372799545160704002e+01,
	1.0070841084922199607011905e+01,
	5.9922447613166942183705192e+00,
	5.8978784056736762299945176e+00,
	3.2822866700678709020367184e+00,
	8.8756430028990417290744307e+00,
	1.0011785496777731986390856e+01,
}

var acos = []complex128{
	(1.0017679804707456328694569 - 2.9138232718554953784519807i),
	(0.03606427612041407369636057 + 2.7358584434576260925091256i),
	(1.6249365462333796703711823 + 2.3159537454335901187730929i),
	(2.0485650849650740120660391 - 3.0795576791204117911123886i),
	(0.29621132089073067282488147 - 3.0007392508200622519398814i),
	(1.0664555914934156601503632 - 2.4872865024796011364747111i),
	(0.48681307452231387690013905 - 2.463655912283054555225301i),
	(0.6116977071277574248407752 - 1.8734458851737055262693056i),
	(1.3649311280370181331184214 + 2.8793528632328795424123832i),
	(2.6189310485682988308904501 - 2.9956543302898767795858704i),
}
var acosh = []complex128{
	(2.9138232718554953784519807 + 1.0017679804707456328694569i),
	(2.7358584434576260925091256 - 0.03606427612041407369636057i),
	(2.3159537454335901187730929 - 1.6249365462333796703711823i),
	(3.0795576791204117911123886 + 2.0485650849650740120660391i),
	(3.0007392508200622519398814 + 0.29621132089073067282488147i),
	(2.4872865024796011364747111 + 1.0664555914934156601503632i),
	(2.463655912283054555225301 + 0.48681307452231387690013905i),
	(1.8734458851737055262693056 + 0.6116977071277574248407752i),
	(2.8793528632328795424123832 - 1.3649311280370181331184214i),
	(2.9956543302898767795858704 + 2.6189310485682988308904501i),
}
var asin = []complex128{
	(0.56902834632415098636186476 + 2.9138232718554953784519807i),
	(1.5347320506744825455349611 - 2.7358584434576260925091256i),
	(-0.054140219438483051139860579 - 2.3159537454335901187730929i),
	(-0.47776875817017739283471738 + 3.0795576791204117911123886i),
	(1.2745850059041659464064402 + 3.0007392508200622519398814i),
	(0.50434073530148095908095852 + 2.4872865024796011364747111i),
	(1.0839832522725827423311826 + 2.463655912283054555225301i),
	(0.9590986196671391943905465 + 1.8734458851737055262693056i),
	(0.20586519875787848611290031 - 2.8793528632328795424123832i),
	(-1.0481347217734022116591284 + 2.9956543302898767795858704i),
}
var asinh = []complex128{
	(2.9113760469415295679342185 + 0.99639459545704326759805893i),
	(2.7441755423994259061579029 - 0.035468308789000500601119392i),
	(-2.2962136462520690506126678 - 1.5144663565690151885726707i),
	(-3.0771233459295725965402455 + 1.0895577967194013849422294i),
	(3.0048366100923647417557027 + 0.29346979169819220036454168i),
	(2.4800059370795363157364643 + 1.0545868606049165710424232i),
	(2.4718773838309585611141821 + 0.47502344364250803363708842i),
	(1.8910743588080159144378396 + 0.56882925572563602341139174i),
	(2.8735426423367341878069406 - 1.362376149648891420997548i),
	(-2.9981750586172477217567878 + 0.5183571985225367505624207i),
}
var atan = []complex128{
	(1.5115747079332741358607654 + 0.091324403603954494382276776i),
	(1.4424504323482602560806727 - 0.0045416132642803911503770933i),
	(-1.5593488703630532674484026 - 0.20163295409248362456446431i),
	(-1.5280619472445889867794105 + 0.081721556230672003746956324i),
	(1.4759909163240799678221039 + 0.028602969320691644358773586i),
	(1.4877353772046548932715555 + 0.14566877153207281663773599i),
	(1.4206983927779191889826 + 0.076830486127880702249439993i),
	(1.3162236060498933364869556 + 0.16031313000467530644933363i),
	(1.5473450684303703578810093 - 0.11064907507939082484935782i),
	(-1.4841462340185253987375812 + 0.049341850305024399493142411i),
}
var atanh = []complex128{
	(0.058375027938968509064640438 + 1.4793488495105334458167782i),
	(0.12977343497790381229915667 - 1.5661009410463561327262499i),
	(-0.010576456067347252072200088 - 1.3743698658402284549750563i),
	(-0.042218595678688358882784918 + 1.4891433968166405606692604i),
	(0.095218997991316722061828397 + 1.5416884098777110330499698i),
	(0.079965459366890323857556487 + 1.4252510353873192700350435i),
	(0.15051245471980726221708301 + 1.4907432533016303804884461i),
	(0.25082072933993987714470373 + 1.392057665392187516442986i),
	(0.022896108815797135846276662 - 1.4609224989282864208963021i),
	(-0.08665624101841876130537396 + 1.5207902036935093480142159i),
}
var conj = []complex128{
	(4.9790119248836735e+00 - 7.7388724745781045e+00i),
	(7.7388724745781045e+00 + 2.7688005719200159e-01i),
	(-2.7688005719200159e-01 + 5.0106036182710749e+00i),
	(-5.0106036182710749e+00 - 9.6362937071984173e+00i),
	(9.6362937071984173e+00 - 2.9263772392439646e+00i),
	(2.9263772392439646e+00 - 5.2290834314593066e+00i),
	(5.2290834314593066e+00 - 2.7279399104360102e+00i),
	(2.7279399104360102e+00 - 1.8253080916808550e+00i),
	(1.8253080916808550e+00 + 8.6859247685756013e+00i),
	(-8.6859247685756013e+00 - 4.9790119248836735e+00i),
}
var cos = []complex128{
	(3.024540920601483938336569e+02 + 1.1073797572517071650045357e+03i),
	(1.192858682649064973252758e-01 + 2.7857554122333065540970207e-01i),
	(7.2144394304528306603857962e+01 - 2.0500129667076044169954205e+01i),
	(2.24921952538403984190541e+03 - 7.317363745602773587049329e+03i),
	(-9.148222970032421760015498e+00 + 1.953124661113563541862227e+00i),
	(-9.116081175857732248227078e+01 - 1.992669213569952232487371e+01i),
	(3.795639179042704640002918e+00 + 6.623513350981458399309662e+00i),
	(-2.9144840732498869560679084e+00 - 1.214620271628002917638748e+00i),
	(-7.45123482501299743872481e+02 + 2.8641692314488080814066734e+03i),
	(-5.371977967039319076416747e+01 + 4.893348341339375830564624e+01i),
}
var cosh = []complex128{
	(8.34638383523018249366948e+00 + 7.2181057886425846415112064e+01i),
	(1.10421967379919366952251e+03 - 3.1379638689277575379469861e+02i),
	(3.051485206773701584738512e-01 - 2.6805384730105297848044485e-01i),
	(-7.33294728684187933370938e+01 + 1.574445942284918251038144e+01i),
	(-7.478643293945957535757355e+03 + 1.6348382209913353929473321e+03i),
	(4.622316522966235701630926e+00 - 8.088695185566375256093098e+00i),
	(-8.544333183278877406197712e+01 + 3.7505836120128166455231717e+01i),
	(-1.934457815021493925115198e+00 + 7.3725859611767228178358673e+00i),
	(-2.352958770061749348353548e+00 - 2.034982010440878358915409e+00i),
	(7.79756457532134748165069e+02 + 2.8549350716819176560377717e+03i),
}
var exp = []complex128{
	(1.669197736864670815125146e+01 + 1.4436895109507663689174096e+02i),
	(2.2084389286252583447276212e+03 - 6.2759289284909211238261917e+02i),
	(2.227538273122775173434327e-01 + 7.2468284028334191250470034e-01i),
	(-6.5182985958153548997881627e-03 - 1.39965837915193860879044e-03i),
	(-1.4957286524084015746110777e+04 + 3.269676455931135688988042e+03i),
	(9.218158701983105935659273e+00 - 1.6223985291084956009304582e+01i),
	(-1.7088175716853040841444505e+02 + 7.501382609870410713795546e+01i),
	(-3.852461315830959613132505e+00 + 1.4808420423156073221970892e+01i),
	(-4.586775503301407379786695e+00 - 4.178501081246873415144744e+00i),
	(4.451337963005453491095747e-05 - 1.62977574205442915935263e-04i),
}
var log = []complex128{
	(2.2194438972179194425697051e+00 + 9.9909115046919291062461269e-01i),
	(2.0468956191154167256337289e+00 - 3.5762575021856971295156489e-02i),
	(1.6130808329853860438751244e+00 - 1.6259990074019058442232221e+00i),
	(2.3851910394823008710032651e+00 + 2.0502936359659111755031062e+00i),
	(2.3096442270679923004800651e+00 + 2.9483213155446756211881774e-01i),
	(1.7904660933974656106951860e+00 + 1.0605860367252556281902109e+00i),
	(1.7745926939841751666177512e+00 + 4.8084556083358307819310911e-01i),
	(1.1885403350045342425648780e+00 + 5.8969634164776659423195222e-01i),
	(2.1833107837679082586772505e+00 - 1.3636647724582455028314573e+00i),
	(2.3037629487273259170991671e+00 + 2.6210913895386013290915234e+00i),
}
var log10 = []complex128{
	(9.6389223745559042474184943e-01 + 4.338997735671419492599631e-01i),
	(8.8895547241376579493490892e-01 - 1.5531488990643548254864806e-02i),
	(7.0055210462945412305244578e-01 - 7.0616239649481243222248404e-01i),
	(1.0358753067322445311676952e+00 + 8.9043121238134980156490909e-01i),
	(1.003065742975330237172029e+00 + 1.2804396782187887479857811e-01i),
	(7.7758954439739162532085157e-01 + 4.6060666333341810869055108e-01i),
	(7.7069581462315327037689152e-01 + 2.0882857371769952195512475e-01i),
	(5.1617650901191156135137239e-01 + 2.5610186717615977620363299e-01i),
	(9.4819982567026639742663212e-01 - 5.9223208584446952284914289e-01i),
	(1.0005115362454417135973429e+00 + 1.1383255270407412817250921e+00i),
}

type ff struct {
	r, theta float64
}

var polar = []ff{
	{9.2022120669932650313380972e+00, 9.9909115046919291062461269e-01},
	{7.7438239742296106616261394e+00, -3.5762575021856971295156489e-02},
	{5.0182478202557746902556648e+00, -1.6259990074019058442232221e+00},
	{1.0861137372799545160704002e+01, 2.0502936359659111755031062e+00},
	{1.0070841084922199607011905e+01, 2.9483213155446756211881774e-01},
	{5.9922447613166942183705192e+00, 1.0605860367252556281902109e+00},
	{5.8978784056736762299945176e+00, 4.8084556083358307819310911e-01},
	{3.2822866700678709020367184e+00, 5.8969634164776659423195222e-01},
	{8.8756430028990417290744307e+00, -1.3636647724582455028314573e+00},
	{1.0011785496777731986390856e+01, 2.6210913895386013290915234e+00},
}
var pow = []complex128{
	(-2.499956739197529585028819e+00 + 1.759751724335650228957144e+00i),
	(7.357094338218116311191939e+04 - 5.089973412479151648145882e+04i),
	(1.320777296067768517259592e+01 - 3.165621914333901498921986e+01i),
	(-3.123287828297300934072149e-07 - 1.9849567521490553032502223e-7i),
	(8.0622651468477229614813e+04 - 7.80028727944573092944363e+04i),
	(-1.0268824572103165858577141e+00 - 4.716844738244989776610672e-01i),
	(-4.35953819012244175753187e+01 + 2.2036445974645306917648585e+02i),
	(8.3556092283250594950239e-01 - 1.2261571947167240272593282e+01i),
	(1.582292972120769306069625e+03 + 1.273564263524278244782512e+04i),
	(6.592208301642122149025369e-08 + 2.584887236651661903526389e-08i),
}
var sin = []complex128{
	(-1.1073801774240233539648544e+03 + 3.024539773002502192425231e+02i),
	(1.0317037521400759359744682e+00 - 3.2208979799929570242818e-02i),
	(-2.0501952097271429804261058e+01 - 7.2137981348240798841800967e+01i),
	(7.3173638080346338642193078e+03 + 2.249219506193664342566248e+03i),
	(-1.964375633631808177565226e+00 - 9.0958264713870404464159683e+00i),
	(1.992783647158514838337674e+01 - 9.11555769410191350416942e+01i),
	(-6.680335650741921444300349e+00 + 3.763353833142432513086117e+00i),
	(1.2794028166657459148245993e+00 - 2.7669092099795781155109602e+00i),
	(2.8641693949535259594188879e+03 + 7.451234399649871202841615e+02i),
	(-4.893811726244659135553033e+01 - 5.371469305562194635957655e+01i),
}
var sinh = []complex128{
	(8.34559353341652565758198e+00 + 7.2187893208650790476628899e+01i),
	(1.1042192548260646752051112e+03 - 3.1379650595631635858792056e+02i),
	(-8.239469336509264113041849e-02 + 9.9273668758439489098514519e-01i),
	(7.332295456982297798219401e+01 - 1.574585908122833444899023e+01i),
	(-7.4786432301380582103534216e+03 + 1.63483823493980029604071e+03i),
	(4.595842179016870234028347e+00 - 8.135290105518580753211484e+00i),
	(-8.543842533574163435246793e+01 + 3.750798997857594068272375e+01i),
	(-1.918003500809465688017307e+00 + 7.4358344619793504041350251e+00i),
	(-2.233816733239658031433147e+00 - 2.143519070805995056229335e+00i),
	(-7.797564130187551181105341e+02 - 2.8549352346594918614806877e+03i),
}
var sqrt = []complex128{
	(2.6628203086086130543813948e+00 + 1.4531345674282185229796902e+00i),
	(2.7823278427251986247149295e+00 - 4.9756907317005224529115567e-02i),
	(1.5397025302089642757361015e+00 - 1.6271336573016637535695727e+00i),
	(1.7103411581506875260277898e+00 + 2.8170677122737589676157029e+00i),
	(3.1390392472953103383607947e+00 + 4.6612625849858653248980849e-01i),
	(2.1117080764822417640789287e+00 + 1.2381170223514273234967850e+00i),
	(2.3587032281672256703926939e+00 + 5.7827111903257349935720172e-01i),
	(1.7335262588873410476661577e+00 + 5.2647258220721269141550382e-01i),
	(2.3131094974708716531499282e+00 - 1.8775429304303785570775490e+00i),
	(8.1420535745048086240947359e-01 + 3.0575897587277248522656113e+00i),
}
var tan = []complex128{
	(-1.928757919086441129134525e-07 + 1.0000003267499169073251826e+00i),
	(1.242412685364183792138948e+00 - 3.17149693883133370106696e+00i),
	(-4.6745126251587795225571826e-05 - 9.9992439225263959286114298e-01i),
	(4.792363401193648192887116e-09 + 1.0000000070589333451557723e+00i),
	(2.345740824080089140287315e-03 + 9.947733046570988661022763e-01i),
	(-2.396030789494815566088809e-05 + 9.9994781345418591429826779e-01i),
	(-7.370204836644931340905303e-03 + 1.0043553413417138987717748e+00i),
	(-3.691803847992048527007457e-02 + 9.6475071993469548066328894e-01i),
	(-2.781955256713729368401878e-08 - 1.000000049848910609006646e+00i),
	(9.4281590064030478879791249e-05 + 9.9999119340863718183758545e-01i),
}
var tanh = []complex128{
	(1.0000921981225144748819918e+00 + 2.160986245871518020231507e-05i),
	(9.9999967727531993209562591e-01 - 1.9953763222959658873657676e-07i),
	(-1.765485739548037260789686e+00 + 1.7024216325552852445168471e+00i),
	(-9.999189442732736452807108e-01 + 3.64906070494473701938098e-05i),
	(9.9999999224622333738729767e-01 - 3.560088949517914774813046e-09i),
	(1.0029324933367326862499343e+00 - 4.948790309797102353137528e-03i),
	(9.9996113064788012488693567e-01 - 4.226995742097032481451259e-05i),
	(1.0074784189316340029873945e+00 - 4.194050814891697808029407e-03i),
	(9.9385534229718327109131502e-01 + 5.144217985914355502713437e-02i),
	(-1.0000000491604982429364892e+00 - 2.901873195374433112227349e-08i),
}

// huge values along the real axis for testing reducePi in Tan
var hugeIn = []complex128{
	1 << 28,
	1 << 29,
	1 << 30,
	1 << 35,
	-1 << 120,
	1 << 240,
	1 << 300,
	-1 << 480,
	1234567891234567 << 180,
	-1234567891234567 << 300,
}

// Results for tanHuge[i] calculated with https://github.com/robpike/ivy
// using 4096 bits of working precision.
var tanHuge = []complex128{
	5.95641897939639421,
	-0.34551069233430392,
	-0.78469661331920043,
	0.84276385870875983,
	0.40806638884180424,
	-0.37603456702698076,
	4.60901287677810962,
	3.39135965054779932,
	-6.76813854009065030,
	-0.76417695016604922,
}

// special cases conform to C99 standard appendix G.6 Complex arithmetic
var inf, nan = math.Inf(1), math.NaN()

var vcAbsSC = []complex128{
	NaN(),
}
var absSC = []float64{
	math.NaN(),
}
var acosSC = []struct {
	in,
	want complex128
}{
	// G.6.1.1
	{complex(zero, zero),
		complex(math.Pi/2, -zero)},
	{complex(-zero, zero),
		complex(math.Pi/2, -zero)},
	{complex(zero, nan),
		complex(math.Pi/2, nan)},
	{complex(-zero, nan),
		complex(math.Pi/2, nan)},
	{complex(1.0, inf),
		complex(math.Pi/2, -inf)},
	{complex(1.0, nan),
		NaN()},
	{complex(-inf, 1.0),
		complex(math.Pi, -inf)},
	{complex(inf, 1.0),
		complex(0.0, -inf)},
	{complex(-inf, inf),
		complex(3*math.Pi/4, -inf)},
	{complex(inf, inf),
		complex(math.Pi/4, -inf)},
	{complex(inf, nan),
		complex(nan, -inf)}, // imaginary sign unspecified
	{complex(-inf, nan),
		complex(nan, inf)}, // imaginary sign unspecified
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		complex(nan, -inf)},
	{NaN(),
		NaN()},
}
var acoshSC = []struct {
	in,
	want complex128
}{
	// G.6.2.1
	{complex(zero, zero),
		complex(zero, math.Pi/2)},
	{complex(-zero, zero),
		complex(zero, math.Pi/2)},
	{complex(1.0, inf),
		complex(inf, math.Pi/2)},
	{complex(1.0, nan),
		NaN()},
	{complex(-inf, 1.0),
		complex(inf, math.Pi)},
	{complex(inf, 1.0),
		complex(inf, zero)},
	{complex(-inf, inf),
		complex(inf, 3*math.Pi/4)},
	{complex(inf, inf),
		complex(inf, math.Pi/4)},
	{complex(inf, nan),
		complex(inf, nan)},
	{complex(-inf, nan),
		complex(inf, nan)},
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		complex(inf, nan)},
	{NaN(),
		NaN()},
}
var asinSC = []struct {
	in,
	want complex128
}{
	// Derived from Asin(z) = -i * Asinh(i * z), G.6 #7
	{complex(zero, zero),
		complex(zero, zero)},
	{complex(1.0, inf),
		complex(0, inf)},
	{complex(1.0, nan),
		NaN()},
	{complex(inf, 1),
		complex(math.Pi/2, inf)},
	{complex(inf, inf),
		complex(math.Pi/4, inf)},
	{complex(inf, nan),
		complex(nan, inf)}, // imaginary sign unspecified
	{complex(nan, zero),
		NaN()},
	{complex(nan, 1),
		NaN()},
	{complex(nan, inf),
		complex(nan, inf)},
	{NaN(),
		NaN()},
}
var asinhSC = []struct {
	in,
	want complex128
}{
	// G.6.2.2
	{complex(zero, zero),
		complex(zero, zero)},
	{complex(1.0, inf),
		complex(inf, math.Pi/2)},
	{complex(1.0, nan),
		NaN()},
	{complex(inf, 1.0),
		complex(inf, zero)},
	{complex(inf, inf),
		complex(inf, math.Pi/4)},
	{complex(inf, nan),
		complex(inf, nan)},
	{complex(nan, zero),
		complex(nan, zero)},
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		complex(inf, nan)}, // sign of real part unspecified
	{NaN(),
		NaN()},
}
var atanSC = []struct {
	in,
	want complex128
}{
	// Derived from Atan(z) = -i * Atanh(i * z), G.6 #7
	{complex(0, zero),
		complex(0, zero)},
	{complex(0, nan),
		NaN()},
	{complex(1.0, zero),
		complex(math.Pi/4, zero)},
	{complex(1.0, inf),
		complex(math.Pi/2, zero)},
	{complex(1.0, nan),
		NaN()},
	{complex(inf, 1),
		complex(math.Pi/2, zero)},
	{complex(inf, inf),
		complex(math.Pi/2, zero)},
	{complex(inf, nan),
		complex(math.Pi/2, zero)},
	{complex(nan, 1),
		NaN()},
	{complex(nan, inf),
		complex(nan, zero)},
	{NaN(),
		NaN()},
}
var atanhSC = []struct {
	in,
	want complex128
}{
	// G.6.2.3
	{complex(zero, zero),
		complex(zero, zero)},
	{complex(zero, nan),
		complex(zero, nan)},
	{complex(1.0, zero),
		complex(inf, zero)},
	{complex(1.0, inf),
		complex(0, math.Pi/2)},
	{complex(1.0, nan),
		NaN()},
	{complex(inf, 1.0),
		complex(zero, math.Pi/2)},
	{complex(inf, inf),
		complex(zero, math.Pi/2)},
	{complex(inf, nan),
		complex(0, nan)},
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		complex(zero, math.Pi/2)}, // sign of real part not specified.
	{NaN(),
		NaN()},
}
var vcConjSC = []complex128{
	NaN(),
}
var conjSC = []complex128{
	NaN(),
}
var cosSC = []struct {
	in,
	want complex128
}{
	// Derived from Cos(z) = Cosh(i * z), G.6 #7
	{complex(zero, zero),
		complex(1.0, -zero)},
	{complex(zero, inf),
		complex(inf, -zero)},
	{complex(zero, nan),
		complex(nan, zero)}, // imaginary sign unspecified
	{complex(1.0, inf),
		complex(inf, -inf)},
	{complex(1.0, nan),
		NaN()},
	{complex(inf, zero),
		complex(nan, -zero)},
	{complex(inf, 1.0),
		NaN()},
	{complex(inf, inf),
		complex(inf, nan)}, // real sign unspecified
	{complex(inf, nan),
		NaN()},
	{complex(nan, zero),
		complex(nan, -zero)}, // imaginary sign unspecified
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		complex(inf, nan)},
	{NaN(),
		NaN()},
}
var coshSC = []struct {
	in,
	want complex128
}{
	// G.6.2.4
	{complex(zero, zero),
		complex(1.0, zero)},
	{complex(zero, inf),
		complex(nan, zero)}, // imaginary sign unspecified
	{complex(zero, nan),
		complex(nan, zero)}, // imaginary sign unspecified
	{complex(1.0, inf),
		NaN()},
	{complex(1.0, nan),
		NaN()},
	{complex(inf, zero),
		complex(inf, zero)},
	{complex(inf, 1.0),
		complex(inf*math.Cos(1.0), inf*math.Sin(1.0))}, // +inf  cis(y)
	{complex(inf, inf),
		complex(inf, nan)}, // real sign unspecified
	{complex(inf, nan),
		complex(inf, nan)},
	{complex(nan, zero),
		complex(nan, zero)}, // imaginary sign unspecified
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		NaN()},
	{NaN(),
		NaN()},
}
var expSC = []struct {
	in,
	want complex128
}{
	// G.6.3.1
	{complex(zero, zero),
		complex(1.0, zero)},
	{complex(-zero, zero),
		complex(1.0, zero)},
	{complex(1.0, inf),
		NaN()},
	{complex(1.0, nan),
		NaN()},
	{complex(inf, zero),
		complex(inf, zero)},
	{complex(-inf, 1.0),
		complex(math.Copysign(0.0, math.Cos(1.0)), math.Copysign(0.0, math.Sin(1.0)))}, // +0 cis(y)
	{complex(inf, 1.0),
		complex(inf*math.Cos(1.0), inf*math.Sin(1.0))}, // +inf  cis(y)
	{complex(-inf, inf),
		complex(zero, zero)}, // real and imaginary sign unspecified
	{complex(inf, inf),
		complex(inf, nan)}, // real sign unspecified
	{complex(-inf, nan),
		complex(zero, zero)}, // real and imaginary sign unspecified
	{complex(inf, nan),
		complex(inf, nan)}, // real sign unspecified
	{complex(nan, zero),
		complex(nan, zero)},
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		NaN()},
	{NaN(),
		NaN()},
}
var vcIsNaNSC = []complex128{
	complex(math.Inf(-1), math.Inf(-1)),
	complex(math.Inf(-1), math.NaN()),
	complex(math.NaN(), math.Inf(-1)),
	complex(0, math.NaN()),
	complex(math.NaN(), 0),
	complex(math.Inf(1), math.Inf(1)),
	complex(math.Inf(1), math.NaN()),
	complex(math.NaN(), math.Inf(1)),
	complex(math.NaN(), math.NaN()),
}
var isNaNSC = []bool{
	false,
	false,
	false,
	true,
	true,
	false,
	false,
	false,
	true,
}

var logSC = []struct {
	in,
	want complex128
}{
	// G.6.3.2
	{complex(zero, zero),
		complex(-inf, zero)},
	{complex(-zero, zero),
		complex(-inf, math.Pi)},
	{complex(1.0, inf),
		complex(inf, math.Pi/2)},
	{complex(1.0, nan),
		NaN()},
	{complex(-inf, 1.0),
		complex(inf, math.Pi)},
	{complex(inf, 1.0),
		complex(inf, 0.0)},
	{complex(-inf, inf),
		complex(inf, 3*math.Pi/4)},
	{complex(inf, inf),
		complex(inf, math.Pi/4)},
	{complex(-inf, nan),
		complex(inf, nan)},
	{complex(inf, nan),
		complex(inf, nan)},
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		complex(inf, nan)},
	{NaN(),
		NaN()},
}
var log10SC = []struct {
	in,
	want complex128
}{
	// derived from Log special cases via Log10(x) = math.Log10E*Log(x)
	{complex(zero, zero),
		complex(-inf, zero)},
	{complex(-zero, zero),
		complex(-inf, float64(math.Log10E)*float64(math.Pi))},
	{complex(1.0, inf),
		complex(inf, float64(math.Log10E)*float64(math.Pi/2))},
	{complex(1.0, nan),
		NaN()},
	{complex(-inf, 1.0),
		complex(inf, float64(math.Log10E)*float64(math.Pi))},
	{complex(inf, 1.0),
		complex(inf, 0.0)},
	{complex(-inf, inf),
		complex(inf, float64(math.Log10E)*float64(3*math.Pi/4))},
	{complex(inf, inf),
		complex(inf, float64(math.Log10E)*float64(math.Pi/4))},
	{complex(-inf, nan),
		complex(inf, nan)},
	{complex(inf, nan),
		complex(inf, nan)},
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		complex(inf, nan)},
	{NaN(),
		NaN()},
}
var vcPolarSC = []complex128{
	NaN(),
}
var polarSC = []ff{
	{math.NaN(), math.NaN()},
}
var vcPowSC = [][2]complex128{
	{NaN(), NaN()},
	{0, NaN()},
}
var powSC = []complex128{
	NaN(),
	NaN(),
}
var sinSC = []struct {
	in,
	want complex128
}{
	// Derived from Sin(z) = -i * Sinh(i * z), G.6 #7
	{complex(zero, zero),
		complex(zero, zero)},
	{complex(zero, inf),
		complex(zero, inf)},
	{complex(zero, nan),
		complex(zero, nan)},
	{complex(1.0, inf),
		complex(inf, inf)},
	{complex(1.0, nan),
		NaN()},
	{complex(inf, zero),
		complex(nan, zero)},
	{complex(inf, 1.0),
		NaN()},
	{complex(inf, inf),
		complex(nan, inf)},
	{complex(inf, nan),
		NaN()},
	{complex(nan, zero),
		complex(nan, zero)},
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		complex(nan, inf)},
	{NaN(),
		NaN()},
}

var sinhSC = []struct {
	in,
	want complex128
}{
	// G.6.2.5
	{complex(zero, zero),
		complex(zero, zero)},
	{complex(zero, inf),
		complex(zero, nan)}, // real sign unspecified
	{complex(zero, nan),
		complex(zero, nan)}, // real sign unspecified
	{complex(1.0, inf),
		NaN()},
	{complex(1.0, nan),
		NaN()},
	{complex(inf, zero),
		complex(inf, zero)},
	{complex(inf, 1.0),
		complex(inf*math.Cos(1.0), inf*math.Sin(1.0))}, // +inf  cis(y)
	{complex(inf, inf),
		complex(inf, nan)}, // real sign unspecified
	{complex(inf, nan),
		complex(inf, nan)}, // real sign unspecified
	{complex(nan, zero),
		complex(nan, zero)},
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		NaN()},
	{NaN(),
		NaN()},
}

var sqrtSC = []struct {
	in,
	want complex128
}{
	// G.6.4.2
	{complex(zero, zero),
		complex(zero, zero)},
	{complex(-zero, zero),
		complex(zero, zero)},
	{complex(1.0, inf),
		complex(inf, inf)},
	{complex(nan, inf),
		complex(inf, inf)},
	{complex(1.0, nan),
		NaN()},
	{complex(-inf, 1.0),
		complex(zero, inf)},
	{complex(inf, 1.0),
		complex(inf, zero)},
	{complex(-inf, nan),
		complex(nan, inf)}, // imaginary sign unspecified
	{complex(inf, nan),
		complex(inf, nan)},
	{complex(nan, 1.0),
		NaN()},
	{NaN(),
		NaN()},
}
var tanSC = []struct {
	in,
	want complex128
}{
	// Derived from Tan(z) = -i * Tanh(i * z), G.6 #7
	{complex(zero, zero),
		complex(zero, zero)},
	{complex(zero, nan),
		complex(zero, nan)},
	{complex(1.0, inf),
		complex(zero, 1.0)},
	{complex(1.0, nan),
		NaN()},
	{complex(inf, 1.0),
		NaN()},
	{complex(inf, inf),
		complex(zero, 1.0)},
	{complex(inf, nan),
		NaN()},
	{complex(nan, zero),
		NaN()},
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		complex(zero, 1.0)},
	{NaN(),
		NaN()},
}
var tanhSC = []struct {
	in,
	want complex128
}{
	// G.6.2.6
	{complex(zero, zero),
		complex(zero, zero)},
	{complex(1.0, inf),
		NaN()},
	{complex(1.0, nan),
		NaN()},
	{complex(inf, 1.0),
		complex(1.0, math.Copysign(0.0, math.Sin(2*1.0)))}, // 1 + i 0 sin(2y)
	{complex(inf, inf),
		complex(1.0, zero)}, // imaginary sign unspecified
	{complex(inf, nan),
		complex(1.0, zero)}, // imaginary sign unspecified
	{complex(nan, zero),
		complex(nan, zero)},
	{complex(nan, 1.0),
		NaN()},
	{complex(nan, inf),
		NaN()},
	{NaN(),
		NaN()},
}

// branch cut continuity checks
// points on each axis at |z| > 1 are checked for one-sided continuity from both the positive and negative side
// all possible branch cuts for the elementary functions are at one of these points

var zero = 0.0
var eps = 1.0 / (1 << 53)

var branchPoints = [][2]complex128{
	{complex(2.0, zero), complex(2.0, eps)},
	{complex(2.0, -zero), complex(2.0, -eps)},
	{complex(-2.0, zero), complex(-2.0, eps)},
	{complex(-2.0, -zero), complex(-2.0, -eps)},
	{complex(zero, 2.0), complex(eps, 2.0)},
	{complex(-zero, 2.0), complex(-eps, 2.0)},
	{complex(zero, -2.0), complex(eps, -2.0)},
	{complex(-zero, -2.0), complex(-eps, -2.0)},
}

// functions borrowed from pkg/math/all_test.go
func tolerance(a, b, e float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}

	// note: b is correct (expected) value, a is actual value.
	// make error tolerance a fraction of b, not a.
	if b != 0 {
		e = e * b
		if e < 0 {
			e = -e
		}
	}
	return d < e
}
func veryclose(a, b float64) bool { return tolerance(a, b, 4e-16) }
func alike(a, b float64) bool {
	switch {
	case a != a && b != b: // math.IsNaN(a) && math.IsNaN(b):
		return true
	case a == b:
		return math.Signbit(a) == math.Signbit(b)
	}
	return false
}

func cTolerance(a, b complex128, e float64) bool {
	d := Abs(a - b)
	if b != 0 {
		e = e * Abs(b)
		if e < 0 {
			e = -e
		}
	}
	return d < e
}
func cSoclose(a, b complex128, e float64) bool { return cTolerance(a, b, e) }
func cVeryclose(a, b complex128) bool          { return cTolerance(a, b, 4e-16) }
func cAlike(a, b complex128) bool {
	var realAlike, imagAlike bool
	if isExact(real(b)) {
		realAlike = alike(real(a), real(b))
	} else {
		// Allow non-exact special cases to have errors in ULP.
		realAlike = veryclose(real(a), real(b))
	}
	if isExact(imag(b)) {
		imagAlike = alike(imag(a), imag(b))
	} else {
		// Allow non-exact special cases to have errors in ULP.
		imagAlike = veryclose(imag(a), imag(b))
	}
	return realAlike && imagAlike
}
func isExact(x float64) bool {
	// Special cases that should match exactly.  Other cases are multiples
	// of Pi that may not be last bit identical on all platforms.
	return math.IsNaN(x) || math.IsInf(x, 0) || x == 0 || x == 1 || x == -1
}

func TestAbs(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Abs(vc[i]); !veryclose(abs[i], f) {
			t.Errorf("Abs(%g) = %g, want %g", vc[i], f, abs[i])
		}
	}
	for i := 0; i < len(vcAbsSC); i++ {
		if f := Abs(vcAbsSC[i]); !alike(absSC[i], f) {
			t.Errorf("Abs(%g) = %g, want %g", vcAbsSC[i], f, absSC[i])
		}
	}
}
func TestAcos(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Acos(vc[i]); !cSoclose(acos[i], f, 1e-14) {
			t.Errorf("Acos(%g) = %g, want %g", vc[i], f, acos[i])
		}
	}
	for _, v := range acosSC {
		if f := Acos(v.in); !cAlike(v.want, f) {
			t.Errorf("Acos(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Acos(Conj(z))  == Conj(Acos(z))
		if f := Acos(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Acos(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
	}
	for _, pt := range branchPoints {
		if f0, f1 := Acos(pt[0]), Acos(pt[1]); !cVeryclose(f0, f1) {
			t.Errorf("Acos(%g) not continuous, got %g want %g", pt[0], f0, f1)
		}
	}
}
func TestAcosh(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Acosh(vc[i]); !cSoclose(acosh[i], f, 1e-14) {
			t.Errorf("Acosh(%g) = %g, want %g", vc[i], f, acosh[i])
		}
	}
	for _, v := range acoshSC {
		if f := Acosh(v.in); !cAlike(v.want, f) {
			t.Errorf("Acosh(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Acosh(Conj(z))  == Conj(Acosh(z))
		if f := Acosh(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Acosh(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}

	}
	for _, pt := range branchPoints {
		if f0, f1 := Acosh(pt[0]), Acosh(pt[1]); !cVeryclose(f0, f1) {
			t.Errorf("Acosh(%g) not continuous, got %g want %g", pt[0], f0, f1)
		}
	}
}
func TestAsin(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Asin(vc[i]); !cSoclose(asin[i], f, 1e-14) {
			t.Errorf("Asin(%g) = %g, want %g", vc[i], f, asin[i])
		}
	}
	for _, v := range asinSC {
		if f := Asin(v.in); !cAlike(v.want, f) {
			t.Errorf("Asin(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Asin(Conj(z))  == Asin(Sinh(z))
		if f := Asin(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Asin(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
		if math.IsNaN(real(v.in)) || math.IsNaN(real(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Asin(-z)  == -Asin(z)
		if f := Asin(-v.in); !cAlike(-v.want, f) && !cAlike(v.in, -v.in) {
			t.Errorf("Asin(%g) = %g, want %g", -v.in, f, -v.want)
		}
	}
	for _, pt := range branchPoints {
		if f0, f1 := Asin(pt[0]), Asin(pt[1]); !cVeryclose(f0, f1) {
			t.Errorf("Asin(%g) not continuous, got %g want %g", pt[0], f0, f1)
		}
	}
}
func TestAsinh(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Asinh(vc[i]); !cSoclose(asinh[i], f, 4e-15) {
			t.Errorf("Asinh(%g) = %g, want %g", vc[i], f, asinh[i])
		}
	}
	for _, v := range asinhSC {
		if f := Asinh(v.in); !cAlike(v.want, f) {
			t.Errorf("Asinh(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Asinh(Conj(z))  == Asinh(Sinh(z))
		if f := Asinh(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Asinh(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
		if math.IsNaN(real(v.in)) || math.IsNaN(real(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Asinh(-z)  == -Asinh(z)
		if f := Asinh(-v.in); !cAlike(-v.want, f) && !cAlike(v.in, -v.in) {
			t.Errorf("Asinh(%g) = %g, want %g", -v.in, f, -v.want)
		}
	}
	for _, pt := range branchPoints {
		if f0, f1 := Asinh(pt[0]), Asinh(pt[1]); !cVeryclose(f0, f1) {
			t.Errorf("Asinh(%g) not continuous, got %g want %g", pt[0], f0, f1)
		}
	}
}
func TestAtan(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Atan(vc[i]); !cVeryclose(atan[i], f) {
			t.Errorf("Atan(%g) = %g, want %g", vc[i], f, atan[i])
		}
	}
	for _, v := range atanSC {
		if f := Atan(v.in); !cAlike(v.want, f) {
			t.Errorf("Atan(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Atan(Conj(z))  == Conj(Atan(z))
		if f := Atan(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Atan(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
		if math.IsNaN(real(v.in)) || math.IsNaN(real(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Atan(-z)  == -Atan(z)
		if f := Atan(-v.in); !cAlike(-v.want, f) && !cAlike(v.in, -v.in) {
			t.Errorf("Atan(%g) = %g, want %g", -v.in, f, -v.want)
		}
	}
	for _, pt := range branchPoints {
		if f0, f1 := Atan(pt[0]), Atan(pt[1]); !cVeryclose(f0, f1) {
			t.Errorf("Atan(%g) not continuous, got %g want %g", pt[0], f0, f1)
		}
	}
}
func TestAtanh(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Atanh(vc[i]); !cVeryclose(atanh[i], f) {
			t.Errorf("Atanh(%g) = %g, want %g", vc[i], f, atanh[i])
		}
	}
	for _, v := range atanhSC {
		if f := Atanh(v.in); !cAlike(v.want, f) {
			t.Errorf("Atanh(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Atanh(Conj(z))  == Conj(Atanh(z))
		if f := Atanh(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Atanh(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
		if math.IsNaN(real(v.in)) || math.IsNaN(real(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Atanh(-z)  == -Atanh(z)
		if f := Atanh(-v.in); !cAlike(-v.want, f) && !cAlike(v.in, -v.in) {
			t.Errorf("Atanh(%g) = %g, want %g", -v.in, f, -v.want)
		}
	}
	for _, pt := range branchPoints {
		if f0, f1 := Atanh(pt[0]), Atanh(pt[1]); !cVeryclose(f0, f1) {
			t.Errorf("Atanh(%g) not continuous, got %g want %g", pt[0], f0, f1)
		}
	}
}
func TestConj(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Conj(vc[i]); !cVeryclose(conj[i], f) {
			t.Errorf("Conj(%g) = %g, want %g", vc[i], f, conj[i])
		}
	}
	for i := 0; i < len(vcConjSC); i++ {
		if f := Conj(vcConjSC[i]); !cAlike(conjSC[i], f) {
			t.Errorf("Conj(%g) = %g, want %g", vcConjSC[i], f, conjSC[i])
		}
	}
}
func TestCos(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Cos(vc[i]); !cSoclose(cos[i], f, 3e-15) {
			t.Errorf("Cos(%g) = %g, want %g", vc[i], f, cos[i])
		}
	}
	for _, v := range cosSC {
		if f := Cos(v.in); !cAlike(v.want, f) {
			t.Errorf("Cos(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Cos(Conj(z))  == Cos(Cosh(z))
		if f := Cos(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Cos(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
		if math.IsNaN(real(v.in)) || math.IsNaN(real(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Cos(-z)  == Cos(z)
		if f := Cos(-v.in); !cAlike(v.want, f) && !cAlike(v.in, -v.in) {
			t.Errorf("Cos(%g) = %g, want %g", -v.in, f, v.want)
		}
	}
}
func TestCosh(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Cosh(vc[i]); !cSoclose(cosh[i], f, 2e-15) {
			t.Errorf("Cosh(%g) = %g, want %g", vc[i], f, cosh[i])
		}
	}
	for _, v := range coshSC {
		if f := Cosh(v.in); !cAlike(v.want, f) {
			t.Errorf("Cosh(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Cosh(Conj(z))  == Conj(Cosh(z))
		if f := Cosh(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Cosh(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
		if math.IsNaN(real(v.in)) || math.IsNaN(real(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Cosh(-z)  == Cosh(z)
		if f := Cosh(-v.in); !cAlike(v.want, f) && !cAlike(v.in, -v.in) {
			t.Errorf("Cosh(%g) = %g, want %g", -v.in, f, v.want)
		}
	}
}
func TestExp(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Exp(vc[i]); !cSoclose(exp[i], f, 1e-15) {
			t.Errorf("Exp(%g) = %g, want %g", vc[i], f, exp[i])
		}
	}
	for _, v := range expSC {
		if f := Exp(v.in); !cAlike(v.want, f) {
			t.Errorf("Exp(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Exp(Conj(z))  == Exp(Cosh(z))
		if f := Exp(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Exp(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
	}
}
func TestIsNaN(t *testing.T) {
	for i := 0; i < len(vcIsNaNSC); i++ {
		if f := IsNaN(vcIsNaNSC[i]); isNaNSC[i] != f {
			t.Errorf("IsNaN(%v) = %v, want %v", vcIsNaNSC[i], f, isNaNSC[i])
		}
	}
}
func TestLog(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Log(vc[i]); !cVeryclose(log[i], f) {
			t.Errorf("Log(%g) = %g, want %g", vc[i], f, log[i])
		}
	}
	for _, v := range logSC {
		if f := Log(v.in); !cAlike(v.want, f) {
			t.Errorf("Log(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Log(Conj(z))  == Conj(Log(z))
		if f := Log(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Log(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
	}
	for _, pt := range branchPoints {
		if f0, f1 := Log(pt[0]), Log(pt[1]); !cVeryclose(f0, f1) {
			t.Errorf("Log(%g) not continuous, got %g want %g", pt[0], f0, f1)
		}
	}
}
func TestLog10(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Log10(vc[i]); !cVeryclose(log10[i], f) {
			t.Errorf("Log10(%g) = %g, want %g", vc[i], f, log10[i])
		}
	}
	for _, v := range log10SC {
		if f := Log10(v.in); !cAlike(v.want, f) {
			t.Errorf("Log10(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Log10(Conj(z))  == Conj(Log10(z))
		if f := Log10(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Log10(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
	}
}
func TestPolar(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if r, theta := Polar(vc[i]); !veryclose(polar[i].r, r) && !veryclose(polar[i].theta, theta) {
			t.Errorf("Polar(%g) = %g, %g want %g, %g", vc[i], r, theta, polar[i].r, polar[i].theta)
		}
	}
	for i := 0; i < len(vcPolarSC); i++ {
		if r, theta := Polar(vcPolarSC[i]); !alike(polarSC[i].r, r) && !alike(polarSC[i].theta, theta) {
			t.Errorf("Polar(%g) = %g, %g, want %g, %g", vcPolarSC[i], r, theta, polarSC[i].r, polarSC[i].theta)
		}
	}
}
func TestPow(t *testing.T) {
	// Special cases for Pow(0, c).
	var zero = complex(0, 0)
	zeroPowers := [][2]complex128{
		{0, 1 + 0i},
		{1.5, 0 + 0i},
		{-1.5, complex(math.Inf(0), 0)},
		{-1.5 + 1.5i, Inf()},
	}
	for _, zp := range zeroPowers {
		if f := Pow(zero, zp[0]); f != zp[1] {
			t.Errorf("Pow(%g, %g) = %g, want %g", zero, zp[0], f, zp[1])
		}
	}
	var a = complex(3.0, 3.0)
	for i := 0; i < len(vc); i++ {
		if f := Pow(a, vc[i]); !cSoclose(pow[i], f, 4e-15) {
			t.Errorf("Pow(%g, %g) = %g, want %g", a, vc[i], f, pow[i])
		}
	}
	for i := 0; i < len(vcPowSC); i++ {
		if f := Pow(vcPowSC[i][0], vcPowSC[i][1]); !cAlike(powSC[i], f) {
			t.Errorf("Pow(%g, %g) = %g, want %g", vcPowSC[i][0], vcPowSC[i][1], f, powSC[i])
		}
	}
	for _, pt := range branchPoints {
		if f0, f1 := Pow(pt[0], 0.1), Pow(pt[1], 0.1); !cVeryclose(f0, f1) {
			t.Errorf("Pow(%g, 0.1) not continuous, got %g want %g", pt[0], f0, f1)
		}
	}
}
func TestRect(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Rect(polar[i].r, polar[i].theta); !cVeryclose(vc[i], f) {
			t.Errorf("Rect(%g, %g) = %g want %g", polar[i].r, polar[i].theta, f, vc[i])
		}
	}
	for i := 0; i < len(vcPolarSC); i++ {
		if f := Rect(polarSC[i].r, polarSC[i].theta); !cAlike(vcPolarSC[i], f) {
			t.Errorf("Rect(%g, %g) = %g, want %g", polarSC[i].r, polarSC[i].theta, f, vcPolarSC[i])
		}
	}
}
func TestSin(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Sin(vc[i]); !cSoclose(sin[i], f, 2e-15) {
			t.Errorf("Sin(%g) = %g, want %g", vc[i], f, sin[i])
		}
	}
	for _, v := range sinSC {
		if f := Sin(v.in); !cAlike(v.want, f) {
			t.Errorf("Sin(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Sin(Conj(z))  == Conj(Sin(z))
		if f := Sin(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Sinh(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
		if math.IsNaN(real(v.in)) || math.IsNaN(real(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Sin(-z)  == -Sin(z)
		if f := Sin(-v.in); !cAlike(-v.want, f) && !cAlike(v.in, -v.in) {
			t.Errorf("Sinh(%g) = %g, want %g", -v.in, f, -v.want)
		}
	}
}
func TestSinh(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Sinh(vc[i]); !cSoclose(sinh[i], f, 2e-15) {
			t.Errorf("Sinh(%g) = %g, want %g", vc[i], f, sinh[i])
		}
	}
	for _, v := range sinhSC {
		if f := Sinh(v.in); !cAlike(v.want, f) {
			t.Errorf("Sinh(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Sinh(Conj(z))  == Conj(Sinh(z))
		if f := Sinh(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Sinh(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
		if math.IsNaN(real(v.in)) || math.IsNaN(real(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Sinh(-z)  == -Sinh(z)
		if f := Sinh(-v.in); !cAlike(-v.want, f) && !cAlike(v.in, -v.in) {
			t.Errorf("Sinh(%g) = %g, want %g", -v.in, f, -v.want)
		}
	}
}
func TestSqrt(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Sqrt(vc[i]); !cVeryclose(sqrt[i], f) {
			t.Errorf("Sqrt(%g) = %g, want %g", vc[i], f, sqrt[i])
		}
	}
	for _, v := range sqrtSC {
		if f := Sqrt(v.in); !cAlike(v.want, f) {
			t.Errorf("Sqrt(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Sqrt(Conj(z)) == Conj(Sqrt(z))
		if f := Sqrt(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Sqrt(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
	}
	for _, pt := range branchPoints {
		if f0, f1 := Sqrt(pt[0]), Sqrt(pt[1]); !cVeryclose(f0, f1) {
			t.Errorf("Sqrt(%g) not continuous, got %g want %g", pt[0], f0, f1)
		}
	}
}
func TestTan(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Tan(vc[i]); !cSoclose(tan[i], f, 3e-15) {
			t.Errorf("Tan(%g) = %g, want %g", vc[i], f, tan[i])
		}
	}
	for _, v := range tanSC {
		if f := Tan(v.in); !cAlike(v.want, f) {
			t.Errorf("Tan(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Tan(Conj(z))  == Conj(Tan(z))
		if f := Tan(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Tan(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
		if math.IsNaN(real(v.in)) || math.IsNaN(real(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Tan(-z)  == -Tan(z)
		if f := Tan(-v.in); !cAlike(-v.want, f) && !cAlike(v.in, -v.in) {
			t.Errorf("Tan(%g) = %g, want %g", -v.in, f, -v.want)
		}
	}
}
func TestTanh(t *testing.T) {
	for i := 0; i < len(vc); i++ {
		if f := Tanh(vc[i]); !cSoclose(tanh[i], f, 2e-15) {
			t.Errorf("Tanh(%g) = %g, want %g", vc[i], f, tanh[i])
		}
	}
	for _, v := range tanhSC {
		if f := Tanh(v.in); !cAlike(v.want, f) {
			t.Errorf("Tanh(%g) = %g, want %g", v.in, f, v.want)
		}
		if math.IsNaN(imag(v.in)) || math.IsNaN(imag(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Tanh(Conj(z))  == Conj(Tanh(z))
		if f := Tanh(Conj(v.in)); !cAlike(Conj(v.want), f) && !cAlike(v.in, Conj(v.in)) {
			t.Errorf("Tanh(%g) = %g, want %g", Conj(v.in), f, Conj(v.want))
		}
		if math.IsNaN(real(v.in)) || math.IsNaN(real(v.want)) {
			// Negating NaN is undefined with regard to the sign bit produced.
			continue
		}
		// Tanh(-z)  == -Tanh(z)
		if f := Tanh(-v.in); !cAlike(-v.want, f) && !cAlike(v.in, -v.in) {
			t.Errorf("Tanh(%g) = %g, want %g", -v.in, f, -v.want)
		}
	}
}

// See issue 17577
func TestInfiniteLoopIntanSeries(t *testing.T) {
	want := Inf()
	if got := Cot(0); got != want {
		t.Errorf("Cot(0): got %g, want %g", got, want)
	}
}

func BenchmarkAbs(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Abs(complex(2.5, 3.5))
	}
}
func BenchmarkAcos(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Acos(complex(2.5, 3.5))
	}
}
func BenchmarkAcosh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Acosh(complex(2.5, 3.5))
	}
}
func BenchmarkAsin(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Asin(complex(2.5, 3.5))
	}
}
func BenchmarkAsinh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Asinh(complex(2.5, 3.5))
	}
}
func BenchmarkAtan(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Atan(complex(2.5, 3.5))
	}
}
func BenchmarkAtanh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Atanh(complex(2.5, 3.5))
	}
}
func BenchmarkConj(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Conj(complex(2.5, 3.5))
	}
}
func BenchmarkCos(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Cos(complex(2.5, 3.5))
	}
}
func BenchmarkCosh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Cosh(complex(2.5, 3.5))
	}
}
func BenchmarkExp(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Exp(complex(2.5, 3.5))
	}
}
func BenchmarkLog(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Log(complex(2.5, 3.5))
	}
}
func BenchmarkLog10(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Log10(complex(2.5, 3.5))
	}
}
func BenchmarkPhase(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Phase(complex(2.5, 3.5))
	}
}
func BenchmarkPolar(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Polar(complex(2.5, 3.5))
	}
}
func BenchmarkPow(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Pow(complex(2.5, 3.5), complex(2.5, 3.5))
	}
}
func BenchmarkRect(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Rect(2.5, 1.5)
	}
}
func BenchmarkSin(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sin(complex(2.5, 3.5))
	}
}
func BenchmarkSinh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sinh(complex(2.5, 3.5))
	}
}
func BenchmarkSqrt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sqrt(complex(2.5, 3.5))
	}
}
func BenchmarkTan(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Tan(complex(2.5, 3.5))
	}
}
func BenchmarkTanh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Tanh(complex(2.5, 3.5))
	}
}
