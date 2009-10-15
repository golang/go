// This file contains the assembly language and machine code for
// Spacewar!, the original PDP-1 video game.  It is downloaded from
// http://spacewar.oversigma.com/sources/sources.zip which has
// the following notice at http://spacewar.oversigma.com/:
//
//	Spacewar! was conceived in 1961 by Martin Graetz, Stephen Russell,
//	and Wayne Wiitanen. It was first realized on the PDP-1 in 1962 by
//	Stephen Russell, Peter Samson, Dan Edwards, and Martin Graetz,
//	together with Alan Kotok, Steve Piner, and Robert A Saunders.
//	Spacewar! is in the public domain, but this credit paragraph must
//	accompany all distributed versions of the program.
//
//	This is the original version! Martin Graetz provided us with a
//	printed version of the source. We typed in in again - it was about
//	40 pages long - and re-assembled it with a PDP-1 assembler written
//	in PERL. The resulting binary runs on a PDP-1 emulator written as
//	a Java applet. The code is extremely faithful to the original. There
//	are only two changes. 1)The spaceships have been made bigger and
//	2) The overall timing has been special cased to deal with varying
//	machine speeds.
//
//	The "a", "s", "d", "f" keys control one of the spaceships. The "k",
//	"l", ";", "'" keys control the other. The controls are spin one
//	way, spin the other, thrust, and fire.
//
//	Barry Silverman
//	Brian Silverman
//	Vadim Gerasimov
//

package main

const spacewarCode = `
-/macro fio-dec system, june 1963
 007652	640500		szm=sza sma-szf
 007652	650500		spq=szm i
 007652	761200		clc=cma+cla-opr
-	define senseswitch A
-	repeat 3, A=A+A
-	szs A
-	term
-	define init A,B
-	law B
-	dap A
-	term
-	define index A,B,C
-	idx A
-	sas B
-	jmp C
-	term
-	define listen
-	cla+cli+clf 1-opr-opr
-	szf i 1
-	jmp .-1
-	tyi
-	term
-	define swap
-	rcl 9s
-	rcl 9s
-	term
-	define load A,B
-	lio (B
-	dio A
-	term
-	define setup A,B
-	law i B
-	dac A
-	term
-	define count A,B
-	isp A
-	jmp B
-	term
-	define move A,B
-	lio A
-	dio B
-	term
-	define clear A,B
-	init .+2, A
-	dzm
-	index .-1, (dzm B+1, .-1
-	term
-/spacewar 3.1  24 sep 62  p1. 1
 000003			3/
 000003	600061		jmp sbf		/ ignore seq. break
 000004	601561		jmp a40
 000005	601556		jmp a1		/ use test word for control, note iot 11 00
-/ interesting and often changed constants
-/symb loc  usual value (all instructions are executed,
-/ and may be replaced by jda or jsp)
 000006		tno,
 000006		6,
 000006	710041		law i 41	/ number of torps + 1
 000007		tvl,
 000007		7,
 000007	675017		sar 4s		/ torpedo velocity
 000010		rlt,
 000010		10,
 000010	710020		law i 20	/ torpedo reload time
 000011		tlf,
 000011		11,
 000011	710140		law i 140	/ torpedo life
 000012		foo,
 000012		12,
 000012	757777		-20000		/ fuel supply
 000013		maa,
 000013		13,
 000013	000010		10		/ spaceship angular acceleration
 000014		sac,
 000014		14,
 000014	675017		sar 4s		/ spaceship acceleration
 000015		str,
 000015		15,
 000015	000001		1		/ star capture radius
 000016		me1,
 000016		16,
 000016	006000		6000		/ collision "radius"
 000017		me2,
 000017		17,
 000017	003000		3000		/ above/2
 000020		ddd,
 000020		20,
 000020	777777		777777		/ 0 to save space for ddt
 000021		the,
 000021		21,
 000021	675777		sar 9s		/ amount of torpedo space warpage
 000022		mhs,
 000022		22,
 000022	710010		law i 10	/ number of hyperspace shots
 000023		hd1,
 000023		23,
 000023	710040		law i 40	/ time in hyperspace before breakout
 000024		hd2,
 000024		24,
 000024	710100		law i 100	/ time in hyperspace breakout
 000025		hd3,
 000025		25,
 000025	710200		law i 200	/ time to recharge hyperfield generator
 000026		hr1,
 000026		26,
 000026	667777		scl 9s		/ scale on hyperspatial displacement
 000027		hr2,
 000027		27,
 000027	667017		scl 4s		/ scale on hyperspatially induced velocity
 000030		hur,
 000030		30,
 000030	040000		40000		/ hyperspatial uncertancy
 000031		ran,
 000031		31,
 000031	000000		0		/ random number
-/ place to build a private control word routine.
-/ it should leave the control word in the io as follows.
-/ high order 4 bits, rotate ccw, rotate cw, (both mean hyperspace)
-/    fire rocket, and fire torpedo. low order 4 bits, same for
-/    other ship. routine is entered by jsp cwg.
 000040			40/
 000040		cwr,
 000040	601672		jmp mg1		/ normally iot 11 control
 000061			. 20/		/ space
-////
-/ routine to flush sequence breaks, if they occur.
 000061		sbf,
 000061	720004		tyi
 000062	220002		lio 2
 000063	200000		lac 0
 000064	720054		lsm
 000065	610001		jmp i 1
-	define xincr X,Y,INS
-	lac Y
-	INS ~ssn
-	dac Y
-	lac X
-	INS ~scn
-	dac X
-	term
-	define yincr X,Y,INS
-	lac Y
-	INS ~scn
-	dac Y
-	lac X
-	-INS+add+sub ~ssn
-	dac X
-	term
-////
-	define dispatch
-	add (a+r
-	dap . 1
-	jmp .
-a,
-	term
-	define dispt A,Y,B
-	repeat 6, B=B+B
-	lio Y
-	dpy-A+B
-	term
-	define scale A,B,C
-	lac A
-	sar B
-	dac C
-	term
-	define diff V,S,QF
-	add i V
-	dac i V
-	xct QF
-	add i S
-	dac i S
-	term
-	define random
-	lac ran
-	rar 1s
-	xor (355760
-	add (355670
-	dac ran
-	term
-	define ranct S,X,C
-	random
-	S
-	X
-	sma
-	cma
-	dac C
-	term
-////
-/sine-cosine subroutine. adams associates
-/calling sequence= number in ac, jda jda sin or jdacos.
-/argument is between q+2 pi, with binary point to right of bit 3.
-/anser has binary point to right of bit 0. time = 2.35 ms.
-	define mult Z
-	jda mpy
-	lac Z
-	term
 000066		cos,
 000066	000000		0
 000067	260142		dap csx
 000070	202760		lac (62210
 000071	400066		add cos
 000072	240074		dac sin
 000073	600077		jmp .+4
 000074		sin,
 000074	000000		0
 000075	260142		dap csx
 000076	200074		lac sin
 000077	640200		spa
 000100		si1,
 000100	402761		add (311040
 000101	422760		sub (62210
 000102	640400		sma
 000103	600143		jmp si2
 000104	402760		add (62210
 000105		si3,
 000105	661003		ral 2s
- 	mult (242763
+000106	170171	    	jda mpy
+000107	202762		lac ZZ11
 000110	240074		dac sin
-	mult sin
+000111	170171	    	jda mpy
+000112	200074		lac ZZ12
 000113	240066		dac cos
-	mult (756103
+000114	170171	    	jda mpy
+000115	202763		lac ZZ13
 000116	402764		add (121312
-	mult cos
+000117	170171	    	jda mpy
+000120	200066		lac ZZ14
 000121	402765		add (532511
-	mult cos
+000122	170171	    	jda mpy
+000123	200066		lac ZZ15
 000124	402766		add (144417
-	mult sin
+000125	170171	    	jda mpy
+000126	200074		lac ZZ16
 000127	667007		scl 3s
 000130	240066		dac cos
 000131	060074		xor sin
 000132	640400		sma
 000133	600141		jmp csx-1
 000134	202767		lac (377777
 000135	220074		lio sin
 000136	642000		spi
 000137	761000		cma
 000140	600142		jmp csx
 000141	200066		lac cos
 000142		csx,
 000142	600142		jmp .
 000143		si2,
 000143	761000		cma
 000144	402760		add (62210
 000145	640400		sma
 000146	600105		jmp si3
 000147	402760		add (62210
 000150	640200		spa
 000151	600154		jmp .+3
 000152	422760		sub (62210
 000153	600105		jmp si3
 000154	422760		sub (62210
 000155	600100		jmp si1
-////
-/bbn multiply subroutine
-/call.. lac one factor, jdy mpy or imp, lac other factor.
 000156		imp,
 000156	000000		0				/returns low 17 bits and sign in ac
 000157	260160		dap im1
 000160		im1,
 000160	100000		xct
 000161	170171		jda mpy
 000162	200156		lac imp
 000163	440160		idx im1
 000164	672001		rir 1s
 000165	673777		rcr 9s
 000166	673777		rcr 9s
 000167	610160		jmp i im1
 000170		mp2,
 000170	000000		0
 000171		mpy,
 000171	000000		0				/return 34 bits and 2 signs
 000172	260200		dap mp1
 000173	200171		lac mpy
 000174	640200		spa
 000175	761000		cma
 000176	673777		rcr 9s
 000177	673777		rcr 9s
 000200		mp1,
 000200	100000		xct
 000201	640200		spa
 000202	761000		cma
 000203	240170		dac mp2
 000204	760200		cla
 000205	540170		mus mp2
+000206	540170		mus mp2
+000207	540170		mus mp2
+000210	540170		mus mp2
+000211	540170		mus mp2
+000212	540170		mus mp2
+000213	540170		mus mp2
+000214	540170		mus mp2
+000215	540170		mus mp2
+000216	540170		mus mp2
+000217	540170		mus mp2
+000220	540170		mus mp2
+000221	540170		mus mp2
+000222	540170		mus mp2
+000223	540170		mus mp2
+000224	540170		mus mp2
+000225	540170		mus mp2
 000226	240170		dac mp2
 000227	100200		xct mp1
 000230	060171		xor mpy
 000231	640400		sma
 000232	600243		jmp mp3
 000233	200170		lac mp2
 000234	761000		cma
 000235	673777		rcr 9s
 000236	673777		rcr 9s
 000237	761000		cma
 000240	673777		rcr 9s
 000241	673777		rcr 9s
 000242	240170		dac mp2
 000243		mp3,
 000243	440200		idx mp1
 000244	200170		lac mp2
 000245	610200		jmp i mp1
-////
-/integer square root
-/input in ac, binary point to right of bit 17, jda sqt
-/answer in ac with binary point between 8 and 9
-/largest input number = 177777
 000246		sqt,
 000246	000000		0
 000247	260260		dap sqx
 000250	710023		law i 23
 000251	240304		dac sq1
 000252	340305		dzm sq2
 000253	220246		lio sqt
 000254	340246		dzm sqt
 000255		sq3,
 000255	460304		isp sq1
 000256	600261		jmp .+3
 000257	200305		lac sq2
 000260		sqx,
 000260	600260		jmp .
 000261	200305		lac sq2
 000262	665001		sal 1s
 000263	240305		dac sq2
 000264	200246		lac sqt
 000265	663003		rcl 2s
 000266	650100		sza i
 000267	600255		jmp sq3
 000270	240246		dac sqt
 000271	200305		lac sq2
 000272	665001		sal 1s
 000273	402770		add (1
 000274	420246		sub sqt
 000275	640500		sma+sza-skip
 000276	600255		jmp sq3
 000277	640200		spa
 000300	761000		cma
 000301	240246		dac sqt
 000302	440305		idx sq2
 000303	600255		jmp sq3
 000304		sq1,
 000304	000000		0
 000305		sq2,
 000305	000000		0
-////
-/bbn divide subroutine
-/calling sequence.. lac hi-dividend, lio lo-dividend, jda dvd, lac divisor.
-/returns quot in ac, rem in io.
 000306		idv,
 000306	000000		0		/integer divide, dividend in ac.
 000307	260317		dap dv1
 000310	200306		lac idv
 000311	677777		scr 9s
 000312	677377		scr 8s
 000313	240315		dac dvd
 000314	600317		jmp dv1
 000315		dvd,
 000315	000000		0
 000316	260317		dap dv1
 000317		dv1,
 000317	100000		xct
 000320	640200		spa
 000321	761000		cma
 000322	240306		dac idv
 000323	200315		lac dvd
 000324	640400		sma
 000325	600334		jmp dv2
 000326	761000		cma
 000327	673777		rcr 9s
 000330	673777		rcr 9s
 000331	761000		cma
 000332	673777		rcr 9s
 000333	673777		rcr 9s
 000334		dv2,
 000334	420306		sub idv
 000335	640400		sma
 000336	600376		jmp dve
 000337	560306		dis idv
+000340	560306		dis idv
+000341	560306		dis idv
+000342	560306		dis idv
+000343	560306		dis idv
+000344	560306		dis idv
+000345	560306		dis idv
+000346	560306		dis idv
+000347	560306		dis idv
+000350	560306		dis idv
+000351	560306		dis idv
+000352	560306		dis idv
+000353	560306		dis idv
+000354	560306		dis idv
+000355	560306		dis idv
+000356	560306		dis idv
+000357	560306		dis idv
+000360	560306		dis idv
 000361	400306		add idv
 000362	320306		dio idv
 000363	764000		cli
 000364	673001		rcr 1s
 000365	220315		lio dvd
 000366	642000		spi
 000367	761000		cma
 000370	240315		dac dvd
 000371	100317		xct dv1
 000372	060315		xor dvd
 000373	673777		rcr 9s
 000374	673777		rcr 9s
 000375	440317		idx dv1
 000376		dve,
 000376	440317		idx dv1
 000377	200306		lac idv
 000400	642000		spi
 000401	761000		cma
 000402	220315		lio dvd
 000403	610317		jmp i dv1
-////
-/outline compiler
-/ac=where to compile to,  call oc
-/ot=address of outline table
-	define	plinst A
-	lac A
-	dac i oc
-	idx oc
-	terminate
-	define comtab A, B
-	plinst A
-	jsp ocs
-	lac B
-	jmp oce
-	terminate
 000404		ocs,
 000404	260411		dap ocz		/puts in swap
 000405	330412		dio i oc
 000406	440412		idx oc
 000407	330412		dio i oc
 000410	440412		idx oc
 000411		ocz,
 000411	600411		jmp .
 000412		oc,
 000412	000000		0
 000413	260554		dap ocx
 000414	210554		lac i ocx
 000415	260434		dap ocg
-	plinst (stf 5
+000416	202771	    	lac ZZ17
+000417	250412		dac i oc
+000420	440412		idx oc
 000421	260555		dap ocm
 000422	440554		idx ocx
 000423		ock,
-	plinst (lac ~sx1
+000423	202772	    	lac ZZ18
+000424	250412		dac i oc
+000425	440412		idx oc
-	plinst (lio ~sy1
+000426	202773	    	lac ZZ19
+000427	250412		dac i oc
+000430	440412		idx oc
 000431	760006		clf 6
 000432		ocj,
-	setup ~occ,6
+000432	710006	    	law i ZZ210
+000433	243112		dac ZZ110
 000434		ocg,
 000434	220434		lio .
 000435		och,
 000435	760200		cla
 000436	663007		rcl 3s
 000437	323113		dio ~oci
 000440	222774		lio (rcl 9s
-	dispatch
+000441	402775	    	add (a11
+000442	260443		dap . 1
+000443	600443		jmp .
+000444		a11,
 000444	760000		opr
 000445	600557		jmp oc1
 000446		oco,
 000446	600602		jmp oc2
 000447		ocq,
 000447	600610		jmp oc3
 000450		ocp,
 000450	600616		jmp oc4
 000451		ocr,
 000451	600624		jmp oc5
 000452	600632		jmp oc6
-////
-	plinst (szf 5		//code
+000453	202776	    	lac ZZ112
+000454	250412		dac i oc
+000455	440412		idx oc
 000456	402777		add (4
 000457	260556		dap ocn
-	plinst ocn
+000460	200556	    	lac ZZ113
+000461	250412		dac i oc
+000462	440412		idx oc
-	plinst (dac ~sx1
+000463	203000	    	lac ZZ114
+000464	250412		dac i oc
+000465	440412		idx oc
-	plinst (dio ~sy1
+000466	203001	    	lac ZZ115
+000467	250412		dac i oc
+000470	440412		idx oc
-	plinst (jmp sq6
+000471	203002	    	lac ZZ116
+000472	250412		dac i oc
+000473	440412		idx oc
-	plinst (clf 5
+000474	203003	    	lac ZZ117
+000475	250412		dac i oc
+000476	440412		idx oc
-	plinst (lac ~scm
+000477	203004	    	lac ZZ118
+000500	250412		dac i oc
+000501	440412		idx oc
-	plinst (cma
+000502	203005	    	lac ZZ119
+000503	250412		dac i oc
+000504	440412		idx oc
-	plinst (dac ~scm
+000505	203006	    	lac ZZ120
+000506	250412		dac i oc
+000507	440412		idx oc
-	plinst (lac ~ssm
+000510	203007	    	lac ZZ121
+000511	250412		dac i oc
+000512	440412		idx oc
-	plinst (cma
+000513	203005	    	lac ZZ122
+000514	250412		dac i oc
+000515	440412		idx oc
-	plinst (dac ~ssm
+000516	203010	    	lac ZZ123
+000517	250412		dac i oc
+000520	440412		idx oc
-	plinst (lac ~csm
+000521	203011	    	lac ZZ124
+000522	250412		dac i oc
+000523	440412		idx oc
-	plinst (lio ~ssd
+000524	203012	    	lac ZZ125
+000525	250412		dac i oc
+000526	440412		idx oc
-	plinst (dac ~ssd
+000527	203013	    	lac ZZ126
+000530	250412		dac i oc
+000531	440412		idx oc
-	plinst (dio ~csm
+000532	203014	    	lac ZZ127
+000533	250412		dac i oc
+000534	440412		idx oc
-	plinst (lac ~ssc
+000535	203015	    	lac ZZ128
+000536	250412		dac i oc
+000537	440412		idx oc
-	plinst (lio ~csn
+000540	203016	    	lac ZZ129
+000541	250412		dac i oc
+000542	440412		idx oc
-	plinst (dac ~csn
+000543	203017	    	lac ZZ130
+000544	250412		dac i oc
+000545	440412		idx oc
-	plinst (dio ~ssc
+000546	203020	    	lac ZZ131
+000547	250412		dac i oc
+000550	440412		idx oc
-	plinst ocm
+000551	200555	    	lac ZZ132
+000552	250412		dac i oc
+000553	440412		idx oc
 000554		ocx,
 000554	600554		jmp .
 000555		ocm,
 000555	600555		jmp .
 000556		ocn,
 000556	600556		jmp .
 000557		oc1,
-	plinst (add ~ssn
+000557	203021	    	lac ZZ133
+000560	250412		dac i oc
+000561	440412		idx oc
 000562	620404		jsp ocs
 000563	203022		lac (sub ~scn
 000564		oce,
 000564	250412		dac i oc
 000565	440412		idx oc
 000566	620404		jsp ocs
-	plinst (ioh
+000567	203023	    	lac ZZ134
+000570	250412		dac i oc
+000571	440412		idx oc
 000572	203024		lac (dpy-4000
 000573		ocd,
 000573	250412		dac i oc
 000574	440412		idx oc
 000575	223113		lio ~oci
-	count ~occ, och
+000576	463112	    	isp ZZ135
+000577	600435		jmp ZZ235
 000600	440434		idx ocg
 000601	600432		jmp ocj
 000602		oc2,
-	comtab (add ~scm, (add ~ssm
-    	plinst ZZ136
+000602	203025	    	lac ZZ137
+000603	250412		dac i oc
+000604	440412		idx oc
+000605	620404		jsp ocs
+000606	203026		lac ZZ236
+000607	600564		jmp oce
 000610		oc3,
-	comtab (add ~ssc, (sub ~csm
-    	plinst ZZ138
+000610	203027	    	lac ZZ139
+000611	250412		dac i oc
+000612	440412		idx oc
+000613	620404		jsp ocs
+000614	203030		lac ZZ238
+000615	600564		jmp oce
 000616		oc4,
-	comtab (sub ~scm, (sub ~ssm
-    	plinst ZZ140
+000616	203031	    	lac ZZ141
+000617	250412		dac i oc
+000620	440412		idx oc
+000621	620404		jsp ocs
+000622	203032		lac ZZ240
+000623	600564		jmp oce
 000624		oc5,
-	comtab (add ~csn, (sub ~ssd
-    	plinst ZZ142
+000624	203033	    	lac ZZ143
+000625	250412		dac i oc
+000626	440412		idx oc
+000627	620404		jsp ocs
+000630	203034		lac ZZ242
+000631	600564		jmp oce
 000632		oc6,
 000632	640006		szf 6
 000633	600642		jmp oc9
 000634	760016		stf 6
-	plinst (dac ~ssa
+000635	203035	    	lac ZZ144
+000636	250412		dac i oc
+000637	440412		idx oc
 000640	203036		lac (dio ~ssi
 000641	600573		jmp ocd
 000642		oc9,
 000642	760006		clf 6
-	plinst (lac ~ssa
+000643	203037	    	lac ZZ145
+000644	250412		dac i oc
+000645	440412		idx oc
 000646	203040		lac (lio ~ssi
 000647	600573		jmp ocd
-////
-/ display a star
-	define starp
-	add ~bx
-	swap
-	add ~by
-	swap
-	ioh
-	dpy-4000
-	terminate
-				/star
 000650		blp,
 000650	260675		dap blx
 000651	640060		szs 60
 000652	600675		jmp blx
-	random
+000653	200031	    	lac ran
+000654	671001		rar 1s
+000655	063041		xor (355760
+000656	403042		add (355670
+000657	240031		dac ran
 000660	671777		rar 9s
 000661	023043		and (add 340
 000662	640200		spa
 000663	062767		xor (377777
 000664	243116		dac ~bx
 000665	200031		lac ran
 000666	661017		ral 4s
 000667	023043		and (add 340
 000670	640200		spa
 000671	062767		xor (377777
 000672	243117		dac ~by
 000673	620676		jsp bpt
 000674	730000		ioh
 000675		blx,
 000675	600675		jmp .
 000676		bpt,
 000676	261117		dap bpx
-	random
+000677	200031	    	lac ran
+000700	671001		rar 1s
+000701	063041		xor (355760
+000702	403042		add (355670
+000703	240031		dac ran
 000704	675777		sar 9s
 000705	675037		sar 5s
 000706	640200		spa
 000707	761000		cma
 000710	665007		sal 3s
 000711	403044		add (bds
 000712	260715		dap bjm
 000713	764206		cla cli clf 6-opr-opr
 000714	724007		dpy-4000
 000715		bjm,
 000715	600715		jmp .
 000716		bds,
-	starp
+000716	403116	    	add ~bx
-	swap
+000717	663777	    	rcl 9s
+000720	663777		rcl 9s
+000721	403117		add ~by
-	swap
+000722	663777	    	rcl 9s
+000723	663777		rcl 9s
+000724	730000		ioh
+000725	724007		dpy-4000
-	starp
+000726	403116	    	add ~bx
-	swap
+000727	663777	    	rcl 9s
+000730	663777		rcl 9s
+000731	403117		add ~by
-	swap
+000732	663777	    	rcl 9s
+000733	663777		rcl 9s
+000734	730000		ioh
+000735	724007		dpy-4000
-	starp
+000736	403116	    	add ~bx
-	swap
+000737	663777	    	rcl 9s
+000740	663777		rcl 9s
+000741	403117		add ~by
-	swap
+000742	663777	    	rcl 9s
+000743	663777		rcl 9s
+000744	730000		ioh
+000745	724007		dpy-4000
-	starp
+000746	403116	    	add ~bx
-	swap
+000747	663777	    	rcl 9s
+000750	663777		rcl 9s
+000751	403117		add ~by
-	swap
+000752	663777	    	rcl 9s
+000753	663777		rcl 9s
+000754	730000		ioh
+000755	724007		dpy-4000
-	starp
+000756	403116	    	add ~bx
-	swap
+000757	663777	    	rcl 9s
+000760	663777		rcl 9s
+000761	403117		add ~by
-	swap
+000762	663777	    	rcl 9s
+000763	663777		rcl 9s
+000764	730000		ioh
+000765	724007		dpy-4000
-	starp
+000766	403116	    	add ~bx
-	swap
+000767	663777	    	rcl 9s
+000770	663777		rcl 9s
+000771	403117		add ~by
-	swap
+000772	663777	    	rcl 9s
+000773	663777		rcl 9s
+000774	730000		ioh
+000775	724007		dpy-4000
-	starp
+000776	403116	    	add ~bx
-	swap
+000777	663777	    	rcl 9s
+001000	663777		rcl 9s
+001001	403117		add ~by
-	swap
+001002	663777	    	rcl 9s
+001003	663777		rcl 9s
+001004	730000		ioh
+001005	724007		dpy-4000
-	starp
+001006	403116	    	add ~bx
-	swap
+001007	663777	    	rcl 9s
+001010	663777		rcl 9s
+001011	403117		add ~by
-	swap
+001012	663777	    	rcl 9s
+001013	663777		rcl 9s
+001014	730000		ioh
+001015	724007		dpy-4000
-	starp
+001016	403116	    	add ~bx
-	swap
+001017	663777	    	rcl 9s
+001020	663777		rcl 9s
+001021	403117		add ~by
-	swap
+001022	663777	    	rcl 9s
+001023	663777		rcl 9s
+001024	730000		ioh
+001025	724007		dpy-4000
-	starp
+001026	403116	    	add ~bx
-	swap
+001027	663777	    	rcl 9s
+001030	663777		rcl 9s
+001031	403117		add ~by
-	swap
+001032	663777	    	rcl 9s
+001033	663777		rcl 9s
+001034	730000		ioh
+001035	724007		dpy-4000
-	starp
+001036	403116	    	add ~bx
-	swap
+001037	663777	    	rcl 9s
+001040	663777		rcl 9s
+001041	403117		add ~by
-	swap
+001042	663777	    	rcl 9s
+001043	663777		rcl 9s
+001044	730000		ioh
+001045	724007		dpy-4000
-	starp
+001046	403116	    	add ~bx
-	swap
+001047	663777	    	rcl 9s
+001050	663777		rcl 9s
+001051	403117		add ~by
-	swap
+001052	663777	    	rcl 9s
+001053	663777		rcl 9s
+001054	730000		ioh
+001055	724007		dpy-4000
-	starp
+001056	403116	    	add ~bx
-	swap
+001057	663777	    	rcl 9s
+001060	663777		rcl 9s
+001061	403117		add ~by
-	swap
+001062	663777	    	rcl 9s
+001063	663777		rcl 9s
+001064	730000		ioh
+001065	724007		dpy-4000
-	starp
+001066	403116	    	add ~bx
-	swap
+001067	663777	    	rcl 9s
+001070	663777		rcl 9s
+001071	403117		add ~by
-	swap
+001072	663777	    	rcl 9s
+001073	663777		rcl 9s
+001074	730000		ioh
+001075	724007		dpy-4000
-	starp
+001076	403116	    	add ~bx
-	swap
+001077	663777	    	rcl 9s
+001100	663777		rcl 9s
+001101	403117		add ~by
-	swap
+001102	663777	    	rcl 9s
+001103	663777		rcl 9s
+001104	730000		ioh
+001105	724007		dpy-4000
-	starp
+001106	403116	    	add ~bx
-	swap
+001107	663777	    	rcl 9s
+001110	663777		rcl 9s
+001111	403117		add ~by
-	swap
+001112	663777	    	rcl 9s
+001113	663777		rcl 9s
+001114	730000		ioh
+001115	724007		dpy-4000
 001116	640006		szf 6
 001117		bpx,
 001117	601117		jmp .
 001120	760016		stf 6
 001121	761000		cma
-	swap
+001122	663777	    	rcl 9s
+001123	663777		rcl 9s
 001124	761000		cma
-	swap
+001125	663777	    	rcl 9s
+001126	663777		rcl 9s
 001127	600715		jmp bjm
-////
-/background display .  3/13/62, prs.
-	define dislis J, Q, B
-	repeat 6, B=B+B
-	clf 5
-	lac flo+r
-	dap fpo+r
-fs,
-	dap fin+r
-	dap fyn+r
-	idx fyn+r
-fin,
-	lac			/lac x
-	sub fpr			/right margin
-	sma
-	jmp fgr+r
-	add (2000
-frr,
-	spq
-fou,
-	jmp fuu+r
-fie,
-	sub (1000
-	sal 8s
-fyn,
-	lio				/lio y
-	dpy-i+B
-	stf 5
-fid,
-	idx fyn+r
-	sad (lio Q+2
-	jmp flp+r
-	sad fpo+r
-	jmp fx+r
-	dap fin+r
-	idx fyn+r
-	jmp fin+r
-fgr,
-	add (2000 -20000
-	jmp frr+r
-fuu,
-	szf 5
-fx,
-	jmp flo+r+1		/return
-	idx flo+r
-	idx flo+r
-	sas (Q+2
-	jmp fid+r
-	law J
-	dac flo+r
-	jmp fid+r
-flp,
-	lac (lio J
-	sad fpo+r
-	jmp fx+r
-	dap fin+r
-	law J+1
-	dap fyn+r
-	jmp fin+r
-fpo,
-	lio
-flo,
-	J
-	terminate
-////
-	define background
-	jsp bck
-	termin
 001130		bck,
 001130	261134		dap bcx
 001131	640040		szs 40
 001132	601134		jmp bcx
 001133	461441		isp bcc
 001134		bcx,
 001134	601134		jmp .
 001135	710002		law i 2
 001136	241441		dac bcc
-	dislis 1j,1q,3
+001137	000006		ZZ398=ZZ398+ZZ398
+001137	000014		ZZ398=ZZ398+ZZ398
+001137	000030		ZZ398=ZZ398+ZZ398
+001137	000060		ZZ398=ZZ398+ZZ398
+001137	000140		ZZ398=ZZ398+ZZ398
+001137	000300		ZZ398=ZZ398+ZZ398
+001137	760005		clf 5
+001140	201214		lac flo98
+001141	261213		dap fpo98
+001142		fs98,
+001142	261145		dap fin98
+001143	261156		dap fyn98
+001144	441156		idx fyn98
+001145		fin98,
+001145	200000		lac
+001146	421443		sub fpr
+001147	640400		sma
+001150	601171		jmp fgr98
+001151	403045		add (2000
+001152		frr98,
+001152	650500		spq
+001153		fou98,
+001153	601173		jmp fuu98
+001154		fie98,
+001154	423046		sub (1000
+001155	665377		sal 8s
+001156		fyn98,
+001156	220000		lio
+001157	720307		dpy-i+ZZ398
+001160	760015		stf 5
+001161		fid98,
+001161	441156		idx fyn98
+001162	503047		sad (lio ZZ298+2
+001163	601204		jmp flp98
+001164	501213		sad fpo98
+001165	601174		jmp fx98
+001166	261145		dap fin98
+001167	441156		idx fyn98
+001170	601145		jmp fin98
+001171		fgr98,
+001171	403050		add (2000 -20000
+001172	601152		jmp frr98
+001173		fuu98,
+001173	640005		szf 5
+001174		fx98,
+001174	601215		jmp flo98+1
+001175	441214		idx flo98
+001176	441214		idx flo98
+001177	523051		sas (ZZ298+2
+001200	601161		jmp fid98
+001201	706000		law ZZ198
+001202	241214		dac flo98
+001203	601161		jmp fid98
+001204		flp98,
+001204	203052		lac (lio ZZ198
+001205	501213		sad fpo98
+001206	601174		jmp fx98
+001207	261145		dap fin98
+001210	706001		law ZZ198+1
+001211	261156		dap fyn98
+001212	601145		jmp fin98
+001213		fpo98,
+001213	220000		lio
+001214		flo98,
+001214	006000		ZZ198
-	dislis 2j,2q,2
+001215	000004		ZZ399=ZZ399+ZZ399
+001215	000010		ZZ399=ZZ399+ZZ399
+001215	000020		ZZ399=ZZ399+ZZ399
+001215	000040		ZZ399=ZZ399+ZZ399
+001215	000100		ZZ399=ZZ399+ZZ399
+001215	000200		ZZ399=ZZ399+ZZ399
+001215	760005		clf 5
+001216	201272		lac flo99
+001217	261271		dap fpo99
+001220		fs99,
+001220	261223		dap fin99
+001221	261234		dap fyn99
+001222	441234		idx fyn99
+001223		fin99,
+001223	200000		lac
+001224	421443		sub fpr
+001225	640400		sma
+001226	601247		jmp fgr99
+001227	403045		add (2000
+001230		frr99,
+001230	650500		spq
+001231		fou99,
+001231	601251		jmp fuu99
+001232		fie99,
+001232	423046		sub (1000
+001233	665377		sal 8s
+001234		fyn99,
+001234	220000		lio
+001235	720207		dpy-i+ZZ399
+001236	760015		stf 5
+001237		fid99,
+001237	441234		idx fyn99
+001240	503053		sad (lio ZZ299+2
+001241	601262		jmp flp99
+001242	501271		sad fpo99
+001243	601252		jmp fx99
+001244	261223		dap fin99
+001245	441234		idx fyn99
+001246	601223		jmp fin99
+001247		fgr99,
+001247	403050		add (2000 -20000
+001250	601230		jmp frr99
+001251		fuu99,
+001251	640005		szf 5
+001252		fx99,
+001252	601273		jmp flo99+1
+001253	441272		idx flo99
+001254	441272		idx flo99
+001255	523054		sas (ZZ299+2
+001256	601237		jmp fid99
+001257	706022		law ZZ199
+001260	241272		dac flo99
+001261	601237		jmp fid99
+001262		flp99,
+001262	203055		lac (lio ZZ199
+001263	501271		sad fpo99
+001264	601252		jmp fx99
+001265	261223		dap fin99
+001266	706023		law ZZ199+1
+001267	261234		dap fyn99
+001270	601223		jmp fin99
+001271		fpo99,
+001271	220000		lio
+001272		flo99,
+001272	006022		ZZ199
-	dislis 3j,3q,1
+001273	000002		ZZ3100=ZZ3100+ZZ3100
+001273	000004		ZZ3100=ZZ3100+ZZ3100
+001273	000010		ZZ3100=ZZ3100+ZZ3100
+001273	000020		ZZ3100=ZZ3100+ZZ3100
+001273	000040		ZZ3100=ZZ3100+ZZ3100
+001273	000100		ZZ3100=ZZ3100+ZZ3100
+001273	760005		clf 5
+001274	201350		lac flo100
+001275	261347		dap fpo100
+001276		fs100,
+001276	261301		dap fin100
+001277	261312		dap fyn100
+001300	441312		idx fyn100
+001301		fin100,
+001301	200000		lac
+001302	421443		sub fpr
+001303	640400		sma
+001304	601325		jmp fgr100
+001305	403045		add (2000
+001306		frr100,
+001306	650500		spq
+001307		fou100,
+001307	601327		jmp fuu100
+001310		fie100,
+001310	423046		sub (1000
+001311	665377		sal 8s
+001312		fyn100,
+001312	220000		lio
+001313	720107		dpy-i+ZZ3100
+001314	760015		stf 5
+001315		fid100,
+001315	441312		idx fyn100
+001316	503056		sad (lio ZZ2100+2
+001317	601340		jmp flp100
+001320	501347		sad fpo100
+001321	601330		jmp fx100
+001322	261301		dap fin100
+001323	441312		idx fyn100
+001324	601301		jmp fin100
+001325		fgr100,
+001325	403050		add (2000 -20000
+001326	601306		jmp frr100
+001327		fuu100,
+001327	640005		szf 5
+001330		fx100,
+001330	601351		jmp flo100+1
+001331	441350		idx flo100
+001332	441350		idx flo100
+001333	523057		sas (ZZ2100+2
+001334	601315		jmp fid100
+001335	706044		law ZZ1100
+001336	241350		dac flo100
+001337	601315		jmp fid100
+001340		flp100,
+001340	203060		lac (lio ZZ1100
+001341	501347		sad fpo100
+001342	601330		jmp fx100
+001343	261301		dap fin100
+001344	706045		law ZZ1100+1
+001345	261312		dap fyn100
+001346	601301		jmp fin100
+001347		fpo100,
+001347	220000		lio
+001350		flo100,
+001350	006044		ZZ1100
-	dislis 4j,4q,0
+001351	000000		ZZ3101=ZZ3101+ZZ3101
+001351	000000		ZZ3101=ZZ3101+ZZ3101
+001351	000000		ZZ3101=ZZ3101+ZZ3101
+001351	000000		ZZ3101=ZZ3101+ZZ3101
+001351	000000		ZZ3101=ZZ3101+ZZ3101
+001351	000000		ZZ3101=ZZ3101+ZZ3101
+001351	760005		clf 5
+001352	201426		lac flo101
+001353	261425		dap fpo101
+001354		fs101,
+001354	261357		dap fin101
+001355	261370		dap fyn101
+001356	441370		idx fyn101
+001357		fin101,
+001357	200000		lac
+001360	421443		sub fpr
+001361	640400		sma
+001362	601403		jmp fgr101
+001363	403045		add (2000
+001364		frr101,
+001364	650500		spq
+001365		fou101,
+001365	601405		jmp fuu101
+001366		fie101,
+001366	423046		sub (1000
+001367	665377		sal 8s
+001370		fyn101,
+001370	220000		lio
+001371	720007		dpy-i+ZZ3101
+001372	760015		stf 5
+001373		fid101,
+001373	441370		idx fyn101
+001374	503061		sad (lio ZZ2101+2
+001375	601416		jmp flp101
+001376	501425		sad fpo101
+001377	601406		jmp fx101
+001400	261357		dap fin101
+001401	441370		idx fyn101
+001402	601357		jmp fin101
+001403		fgr101,
+001403	403050		add (2000 -20000
+001404	601364		jmp frr101
+001405		fuu101,
+001405	640005		szf 5
+001406		fx101,
+001406	601427		jmp flo101+1
+001407	441426		idx flo101
+001410	441426		idx flo101
+001411	523062		sas (ZZ2101+2
+001412	601373		jmp fid101
+001413	706306		law ZZ1101
+001414	241426		dac flo101
+001415	601373		jmp fid101
+001416		flp101,
+001416	203063		lac (lio ZZ1101
+001417	501425		sad fpo101
+001420	601406		jmp fx101
+001421	261357		dap fin101
+001422	706307		law ZZ1101+1
+001423	261370		dap fyn101
+001424	601357		jmp fin101
+001425		fpo101,
+001425	220000		lio
+001426		flo101,
+001426	006306		ZZ1101
 001427	461442		isp bkc
 001430	601134		jmp bcx
 001431	710020		law i 20
 001432	241442		dac bkc
 001433	710001		law i 1
 001434	401443		add fpr
 001435	640200		spa
 001436	403064		add (20000
 001437	241443		dac fpr
 001440	601134		jmp bcx
 001441		bcc,
 001441	000000		0
 001442		bkc,
 001442	000000		0
 001443		fpr,
 001443	010000		10000
-////
-/spacewar 3.1  24 sep 62  pt. 2
-/main control for spaceships
 001444	000030		nob=30			/total number of colliding objects
 001444		ml0,
-	load ~mtc, -4000	/delay for loop
+001444	223065	    	lio (ZZ2102
+001445	323120		dio ZZ1102
-	init ml1, mtb		/loc of calc routines
+001446	703365	    	law ZZ2103
+001447	261703		dap ZZ1103
 001450	403066		add (nob
 001451	261737		dap mx1			/x
 001452	003415		nx1=mtb nob
 001452	403066		add (nob
 001453	261747		dap my1			/y
 001454	003445		ny1=nx1 nob
 001454	403066		add (nob
 001455	261772		dap ma1			/ count for length of explosion or torp
 001456	003475		na1=ny1 nob
 001456	403066		add (nob
 001457	262006		dap mb1			/ count of instructions taken by calc routine
 001460	003525		nb1=na1 nob
 001460	403066		add (nob
 001461	243121		dac ~mdx		/ dx
 001462	003555		ndx=nb1 nob
 001462	403066		add (nob
 001463	243122		dac ~mdy		/ dy
 001464	003605		ndy=ndx nob
 001464	403066		add (nob
 001465	262327		dap mom			/angular velocity
 001466	003635		nom=ndy nob
 001466	403067		add (2
 001467	262343		dap mth			/ angle
 001470	003637		nth=nom 2
 001470	403067		add (2
 001471	243123		dac ~mfu		/fuel
 001472	003641		nfu=nth 2
 001472	403067		add (2
 001473	243124		dac ~mtr		/ no torps remaining
 001474	003643		ntr=nfu 2
 001474	403067		add (2
 001475	261732		dap mot			/ outline of spaceship
 001476	003645		not=ntr 2
 001476	403067		add (2
 001477	262577		dap mco			/ old control word
 001500	003647		nco=not 2
 001500	403067		add (2
 001501	243125		dac ~mh1
 001502	003651		nh1=nco 2
 001502	403067		add (2
 001503	243126		dac ~mh2
 001504	003653		nh2=nh1 2
 001504	403067		add (2
 001505	243127		dac ~mh3
 001506	003655		nh3=nh2 2
 001506	403067		add (2
 001507	243130		dac ~mh4
 001510	003657		nh4=nh3 2
 001510	003661		nnn=nh4 2
-////
 001510	702310		law ss1
 001511	063365		xor mtb
 001512	640100		sza
 001513	601534		jmp mdn
 001514	702314		law ss2
 001515	063366		xor mtb 1
 001516	640100		sza
 001517	601534		jmp mdn
 001520	700001		law 1			/ test if both ships out of torps
 001521	403643		add ntr
 001522	640200		spa
 001523	601530		jmp md1
 001524	700001		law 1
 001525	403644		add ntr 1
 001526	650200		spa i
 001527	601534		jmp mdn
 001530		md1,
 001530	100011		xct tlf			/ restart delay is 2x torpedo life
 001531	665001		sal 1s
 001532	243131		dac ~ntd
 001533	601703		jmp ml1
 001534		mdn,
-	count ~ntd,ml1
+001534	463131	    	isp ZZ1104
+001535	601703		jmp ZZ2104
 001536	760011		stf 1
 001537	760012		stf 2
 001540	702310		law ss1
 001541	063365		xor mtb
 001542	640100		sza
 001543	760001		clf 1
 001544	650100		sza i
 001545	443132		idx ~1sc
 001546	702314		law ss2
 001547	063366		xor mtb 1
 001550	640100		sza
 001551	760002		clf 2
 001552	650100		sza i
 001553	443133		idx ~2sc
 001554	760002		clf 2
 001555	601564		jmp a
-////
 001556		a1,
 001556	701676		law mg2			/ test word control
 001557	243134		dac ~cwg
 001560	601564		jmp a
 001561		a40,
 001561	700040		law cwr			/ here from start at 4
 001562	243134		dac ~cwg
 001563	601613		jmp a6
 001564		a,
 001564	203135		lac ~gct
 001565	640400		sma
 001566	601576		jmp a5
-	count ~gct, a5
+001567	463135	    	isp ZZ1105
+001570	601576		jmp ZZ2105
 001571	203132		lac ~1sc
 001572	523133		sas ~2sc
 001573	601602		jmp a4
 001574	710001		law i 1
 001575	243135		dac ~gct
 001576		a5,
 001576	762200		lat
 001577	023070		and (40
 001600	650100		sza i
 001601	601621		jmp a2
 001602		a4,
 001602	203132		lac ~1sc
 001603	223133		lio ~2sc
 001604	760400		hlt
 001605	762200		lat
 001606	023070		and (40
 001607	640100		sza
 001610	601621		jmp a2
 001611	343132		dzm ~1sc
 001612	343133		dzm ~2sc
 001613		a6,
 001613	762200		lat
 001614	671077		rar 6s
 001615	023071		and (37
 001616	640100		sza
 001617	761000		cma
 001620	243135		dac ~gct
 001621		a2,
-	clear mtb, nnn-1		/ clear out all tables
-    	init .+2, ZZ1106
+001621	703365	    	law ZZ2107
+001622	261623		dap ZZ1107
+001623	340000		dzm
-	index .-1, (dzm ZZ2106+1, .-1
+001624	441623	    	idx ZZ1108
+001625	523072		sas ZZ2108
+001626	601623		jmp ZZ3108
 001627	702310		law ss1
 001630	243365		dac mtb
 001631	702314		law ss2
 001632	243366		dac mtb 1
 001633	203073		lac (200000
 001634	243415		dac nx1
 001635	243445		dac ny1
 001636	761000		cma
 001637	243416		dac nx1 1
 001640	243446		dac ny1 1
 001641	203074		lac (144420
 001642	243637		dac nth
-////
 001643	703661		law nnn					/ start of outline problem
 001644	243645		dac not
 001645	220020		lio ddd
 001646	652000		spi i
 001647	601652		jmp a3
 001650	170412		jda oc
 001651	002735		ot1
 001652		a3,
 001652	243646		dac not 1
 001653	170412		jda oc
 001654	002746		ot2
 001655	100006		xct tno
 001656	243643		dac ntr
 001657	243644		dac ntr 1
 001660	200012		lac foo
 001661	243641		dac nfu
 001662	243642		dac nfu 1
 001663	702000		law 2000
 001664	243525		dac nb1
 001665	243526		dac nb1 1
 001666	100022		xct mhs
 001667	243653		dac nh2
 001670	243654		dac nh2 1
 001671	601444		jmp ml0
-/ control word get routines
 001672		mg1,
 001672	261675		dap mg3
 001673	764000		cli
 001674	720011		iot 11
 001675		mg3,
 001675	601675		jmp .
 001676		mg2,
 001676	261702		dap mg4
 001677	762200		lat
-	swap
+001700	663777	    	rcl 9s
+001701	663777		rcl 9s
 001702		mg4,
 001702	601702		jmp .
-////
 001703		ml1,
 001703	201703		lac .			/ 1st control word
 001704	650100		sza i			/ zero if not active
 001705	602011		jmp mq1			/ not active
-	swap
+001706	663777	    	rcl 9s
+001707	663777		rcl 9s
 001710	443136		idx ~moc
 001711	642000		spi
 001712	602003		jmp mq4
 001713	700001		law 1
 001714	401703		add ml1
 001715	261734		dap ml2
 001716	700001		law 1
 001717	401737		add mx1
 001720	261740		dap mx2
 001721	700001		law 1
 001722	401747		add my1
 001723	261750		dap my2
 001724	700001		law 1
 001725	401772		add ma1
 001726	261773		dap ma2
 001727	700001		law 1
 001730	402006		add mb1
 001731	261766		dap mb2
 001732		mot,
 001732	201732		lac .
 001733	262530		dap sp5
 001734		ml2,
 001734	201734		lac .			/ 2nd control word
 001735	650500		spq			/ can it collide?
 001736	601774		jmp mq2			/ no
 001737		mx1,
 001737	201737		lac .			/ calc if collision
 001740		mx2,
 001740	421740		sub .			/ delta x
 001741	640200		spa			/ take abs value
 001742	761000		cma
 001743	243137		dac ~mt1
 001744	420016		sub me1			/ < epsilon ?
 001745	640400		sma
 001746	601774		jmp mq2			/ no
 001747		my1,
 001747	201747		lac .
 001750		my2,
 001750	421750		sub .
 001751	640200		spa
 001752	761000		cma
 001753	420016		sub me1			/ < epsilon ?
 001754	640400		sma
 001755	601774		jmp mq2			/ no
 001756	403137		add ~mt1
 001757	420017		sub me2
 001760	640400		sma
 001761	601774		jmp mq2
 001762	203103		lac (mex 400000	/ yes, explode
 001763	251703		dac i ml1		/ replace calc routine with explosion
 001764	251734		dac i ml2
 001765	212006		lac i mb1		/ duration of explosion
 001766		mb2,
 001766	401766		add .
 001767	761000		cma
 001770	675377		sar 8s
 001771	402770		add (1
 001772		ma1,
 001772	241772		dac .
 001773		ma2,
 001773	241773		dac .
 001774		mq2,
 001774	441740		idx mx2			/ end of comparion loop
 001775	441750		idx my2
 001776	441773		idx ma2
 001777	441766		idx mb2
-	index ml2, (lac mtb nob, ml2
+002000	441734	    	idx ZZ1111
+002001	523075		sas ZZ2111
+002002	601734		jmp ZZ3111
-////
 002003		mq4,
 002003	211703		lac i ml1		/ routine for calculating spaceship
 002004	262005		dap . 1			/ or other object and displaying it
 002005	622005		jsp .
 002006		mb1,
 002006	202006		lac .			/ alter count of number of instructions
 002007	403120		add ~mtc
 002010	243120		dac ~mtc
 002011		mq1,
 002011	441737		idx mx1			/ end of comparison and display loop
 002012	441747		idx my1
 002013	441772		idx ma1
 002014	442006		idx mb1
 002015	443121		idx ~mdx
 002016	443122		idx ~mdy
 002017	442327		idx mom
 002020	442343		idx mth
 002021	443140		idx ~mas
 002022	443123		idx ~mfu
 002023	443124		idx ~mtr
 002024	441732		idx mot
 002025	442577		idx mco
 002026	443125		idx ~mh1
 002027	443126		idx ~mh2
 002030	443127		idx ~mh3
 002031	443130		idx ~mh4
-	index ml1, (lac mtb nob-1, ml1
+002032	441703	    	idx ZZ1112
+002033	523076		sas ZZ2112
+002034	601703		jmp ZZ3112
 002035	211703		lac i ml1		/ display and compute last point
 002036	650100		sza i			/ if active
 002037	602045		jmp mq3
 002040	262041		dap . 1
 002041	622041		jsp .
 002042	212006		lac i mb1
 002043	403120		add ~mtc
 002044	243120		dac ~mtc
 002045		mq3,
-	background		/ display stars of the heavens
+002045	621130	    	jsp bck
 002046	620650		jsp blp			/ display massive star
-	count ~mtc, .	/ use the rest of time of main loop
+002047	463120	    	isp ZZ1114
+002050	602047		jmp ZZ2114
 002051	601444		jmp ml0			/ repeat whole works
-////
-/ misc calculation routines
-	/ explosion
 002052		mex,
 002052	262133		dap mxr
 002053	760200		cla
-	diff ~mdx, mx1, (sar 3s
+002054	413121	    	add i ZZ1115
+002055	253121		dac i ZZ1115
+002056	103077		xct ZZ3115
+002057	411737		add i ZZ2115
+002060	251737		dac i ZZ2115
 002061	760200		cla
-	diff ~mdy, my1, (sar 3s
+002062	413122	    	add i ZZ1116
+002063	253122		dac i ZZ1116
+002064	103077		xct ZZ3116
+002065	411747		add i ZZ2116
+002066	251747		dac i ZZ2116
 002067	702134		law ms2
 002070	262117		dap msh
 002071	212006		lac i mb1		/ time involved
 002072	765000		cma cli-opr
 002073	675007		sar 3s
 002074	243141		dac ~mxc
 002075		ms1,
 002075	423100		sub (140
 002076	640400		sma
 002077	442117		idx msh
 002100		mz1,
-	random
+002100	200031	    	lac ran
+002101	671001		rar 1s
+002102	063041		xor (355760
+002103	403042		add (355670
+002104	240031		dac ran
 002105	023101		and (777
 002106	043102		ior (scl
 002107	242120		dac mi1
-	random
+002110	200031	    	lac ran
+002111	671001		rar 1s
+002112	063041		xor (355760
+002113	403042		add (355670
+002114	240031		dac ran
 002115	677777		scr 9s
 002116	676777		sir 9s
 002117		msh,
 002117	102117		xct .
 002120		mi1,
 002120	760400		hlt
 002121	411747		add i my1
-	swap
+002122	663777	    	rcl 9s
+002123	663777		rcl 9s
 002124	411737		add i mx1
 002125	720307		dpy-i 300
-	count ~mxc, mz1
+002126	463141	    	isp ZZ1120
+002127	602100		jmp ZZ2120
-	count i ma1, mxr
+002130	471772	    	isp ZZ1121
+002131	602133		jmp ZZ2121
 002132	351703		dzm i ml1
 002133		mxr,
 002133	602133		jmp .
 002134		ms2,
 002134	677001		scr 1s
 002135	677007		scr 3s
-/ torpedo calc routine
 002136		tcr,
 002136	262167		dap trc
-	count i ma1, tc1
+002137	471772	    	isp ZZ1122
+002140	602146		jmp ZZ2122
 002141	203103		lac (mex 400000
 002142	251703		dac i ml1
 002143	710002		law i 2
 002144	251772		dac i ma1
 002145	602167		jmp trc
 002146		tc1,
 002146	211737		lac i mx1
 002147	675777		sar 9s
 002150	100021		xct the
-	diff ~mdy, my1, (sar 3s
+002151	413122	    	add i ZZ1123
+002152	253122		dac i ZZ1123
+002153	103077		xct ZZ3123
+002154	411747		add i ZZ2123
+002155	251747		dac i ZZ2123
 002156	675777		sar 9s
 002157	100021		xct the
-	diff ~mdx, mx1, (sar 3s
+002160	413121	    	add i ZZ1124
+002161	253121		dac i ZZ1124
+002162	103077		xct ZZ3124
+002163	411737		add i ZZ2124
+002164	251737		dac i ZZ2124
-	dispt i, i my1, 1
+002165	000002		ZZ3125=ZZ3125+ZZ3125
+002165	000004		ZZ3125=ZZ3125+ZZ3125
+002165	000010		ZZ3125=ZZ3125+ZZ3125
+002165	000020		ZZ3125=ZZ3125+ZZ3125
+002165	000040		ZZ3125=ZZ3125+ZZ3125
+002165	000100		ZZ3125=ZZ3125+ZZ3125
+002165	231747		lio ZZ2125
+002166	720107		dpy-ZZ1125+ZZ3125
 002167		trc,
 002167	602167		jmp .
-////
-/ hyperspace routines
-/ this routine handles a non-colliding ship invisibly
-/ in hyperspace
 002170		hp1,
 002170	262245		dap hp2
-	count i ma1, hp2
+002171	471772	    	isp ZZ1126
+002172	602245		jmp ZZ2126
 002173	702246		law hp3				/ next step
 002174	251703		dac i ml1
 002175	700007		law 7
 002176	252006		dac i mb1
-	random
+002177	200031	    	lac ran
+002200	671001		rar 1s
+002201	063041		xor (355760
+002202	403042		add (355670
+002203	240031		dac ran
 002204	677777		scr 9s
 002205	676777		sir 9s
 002206	100026		xct hr1
 002207	411737		add i mx1
 002210	251737		dac i mx1
-	swap
+002211	663777	    	rcl 9s
+002212	663777		rcl 9s
 002213	411747		add i my1
 002214	251747		dac i my1
-	random
+002215	200031	    	lac ran
+002216	671001		rar 1s
+002217	063041		xor (355760
+002220	403042		add (355670
+002221	240031		dac ran
 002222	677777		scr 9s
 002223	676777		sir 9s
 002224	100027		xct hr2
 002225	253122		dac i ~mdy
 002226	333121		dio i ~mdx
-	setup ~hpt,3
+002227	710003	    	law i ZZ2130
+002230	243142		dac ZZ1130
 002231	200031		lac ran
 002232	252343		dac i mth
 002233		hp4,
 002233	212343		lac i mth
 002234	640400		sma
 002235	422761		sub (311040
 002236	640200		spa
 002237	402761		add (311040
 002240	252343		dac i mth
-	count ~hpt,hp4
+002241	463142	    	isp ZZ1131
+002242	602233		jmp ZZ2131
 002243	100024		xct hd2
 002244	251772		dac i ma1
 002245		hp2,
 002245	602245		jmp .
-/ this routine handles a ship breaking out of
-/ hyperspace
 002246		hp3,
 002246	262307		dap hp5
-	count i ma1,hp6
+002247	471772	    	isp ZZ1132
+002250	602304		jmp ZZ2132
 002251	213125		lac i ~mh1
 002252	251703		dac i ml1
 002253	702000		law 2000
 002254	252006		dac i mb1
-	count i ~mh2,hp7
+002255	473126	    	isp ZZ1133
+002256	602260		jmp ZZ2133
 002257	353126		dzm i ~mh2
-////
 002260		hp7,
 002260	100025		xct hd3
 002261	253127		dac i ~mh3
 002262	213130		lac i ~mh4
 002263	400030		add hur
 002264	253130		dac i ~mh4
-	random
+002265	200031	    	lac ran
+002266	671001		rar 1s
+002267	063041		xor (355760
+002270	403042		add (355670
+002271	240031		dac ran
 002272	043104		ior (400000
 002273	413130		add i ~mh4
 002274	640200		spa
 002275	602307		jmp hp5
 002276	203103		lac (mex 400000
 002277	251703		dac i ml1
 002300	710010		law i 10
 002301	251772		dac i ma1
 002302	702000		law 2000
 002303	252006		dac i mb1
 002304		hp6,
 002304	211737		lac i mx1
-	dispt i, i my1, 2
+002305	000004		ZZ3135=ZZ3135+ZZ3135
+002305	000010		ZZ3135=ZZ3135+ZZ3135
+002305	000020		ZZ3135=ZZ3135+ZZ3135
+002305	000040		ZZ3135=ZZ3135+ZZ3135
+002305	000100		ZZ3135=ZZ3135+ZZ3135
+002305	000200		ZZ3135=ZZ3135+ZZ3135
+002305	231747		lio ZZ2135
+002306	720207		dpy-ZZ1135+ZZ3135
 002307		hp5,
 002307	602307		jmp .
-////
-/ spaceship calc
 002310		ss1,
 002310	262713		dap srt			/ first spaceship
 002311	633134		jsp i ~cwg
 002312	323143		dio ~scw
 002313	602320		jmp sr0
 002314		ss2,
 002314	262713		dap srt
 002315	633134		jsp i ~cwg
 002316	672017		rir 4s
 002317	323143		dio ~scw
 002320		sr0,
 002320		sc1,
 002320	223143		lio ~scw		/control word
 002321	760206		clf 6 cla-opr		/update angle
 002322	642000		spi
 002323	400013		add maa
 002324	662001		ril 1s
 002325	642000		spi
 002326	420013		sub maa
 002327		mom,
 002327	402327		add .
 002330	252327		dac i mom
 002331	640010		szs 10
 002332	602335		jmp sr8
 002333	352327		dzm i mom
 002334	661177		ral 7s
 002335		sr8,
 002335	662001		ril 1s
 002336	642000		spi
 002337	760016		stf 6
 002340	233123		lio i ~mfu
 002341	652000		spi i
 002342	760006		clf 6
 002343		mth,
 002343	402343		add .
 002344	640400		sma
 002345	422761		sub (311040
 002346	640200		spa
 002347	402761		add (311040
 002350	252343		dac i mth
 002351	170074		jda sin
 002352	243144		dac ~sn
 002353	343116		dzm ~bx
 002354	343117		dzm ~by
 002355	640060		szs 60
 002356	602430		jmp bsg
 002357	211737		lac i mx1
 002360	675777		sar 9s
 002361	675003		sar 2s
 002362	243145		dac ~t1
 002363	170156		jda imp
 002364	203145		lac ~t1
 002365	243146		dac ~t2
 002366	211747		lac i my1
-////
 002367	675777		sar 9s
 002370	675003		sar 2s
 002371	243145		dac ~t1
 002372	170156		jda imp
 002373	203145		lac ~t1
 002374	403146		add ~t2
 002375	420015		sub str
 002376	650500		sma i sza-skp
 002377	602714		jmp poh
 002400	400015		add str
 002401	243145		dac ~t1
 002402	170246		jda sqt
 002403	675777		sar 9s
 002404	170171		jda mpy
 002405	203145		lac ~t1
 002406	677003		scr 2s
 002407	650020		szs i 20		/ switch 2 for light star
 002410	677003		scr 2s
 002411	640100		sza
 002412	602430		jmp bsg
 002413	323145		dio ~t1
 002414	211737		lac i mx1
 002415	761000		cma
 002416	170306		jda idv
 002417	203145		lac ~t1
 002420	760000		opr
 002421	243116		dac ~bx
 002422	211747		lac i my1
 002423	761000		cma
 002424	170306		jda idv
 002425	203145		lac ~t1
 002426	760000		opr
 002427	243117		dac ~by
 002430		bsg,
 002430	760200		cla
 002431	513123		sad i ~mfu
 002432	760006		clf 6
 002433	212343		lac i mth
 002434	170066		jda cos
 002435	243147		dac ~cs
 002436	675777		sar 9s
 002437	100014		xct sac
 002440	650006		szf i 6
 002441	760200		cla
 002442	403117		add ~by
-	diff ~mdy, my1, (sar 3s
+002443	413122	    	add i ZZ1136
+002444	253122		dac i ZZ1136
+002445	103077		xct ZZ3136
+002446	411747		add i ZZ2136
+002447	251747		dac i ZZ2136
 002450	203144		lac ~sn
 002451	675777		sar 9s
 002452	100014		xct sac
 002453	761000		cma
 002454	650006		szf i 6
 002455	760200		cla
 002456	403116		add ~bx
-	diff ~mdx, mx1, (sar 3s
+002457	413121	    	add i ZZ1137
+002460	253121		dac i ZZ1137
+002461	103077		xct ZZ3137
+002462	411737		add i ZZ2137
+002463	251737		dac i ZZ2137
 002464		sp1,
-	scale ~sn, 5s, ~ssn
+002464	203144	    	lac ZZ1138
+002465	675037		sar ZZ2138
+002466	243150		dac ZZ3138
 002467		sp2,
-	scale ~cs, 5s, ~scn
+002467	203147	    	lac ZZ1139
+002470	675037		sar ZZ2139
+002471	243114		dac ZZ3139
 002472	211737		lac i mx1
-////
 002473	423150		sub ~ssn
 002474	243151		dac ~sx1
 002475	423150		sub ~ssn
 002476	243152		dac ~stx
 002477	211747		lac i my1
 002500	403114		add ~scn
 002501	243153		dac ~sy1
 002502	403114		add ~scn
 002503	243154		dac ~sty
-/ Modified for Smaller Laptop screens - BDS
-//	scale ~sn, 9s, ~ssn
-//	scale ~cs, 9s, ~scn
-	scale ~sn, 8s, ~ssn
+002504	203144	    	lac ZZ1140
+002505	675377		sar ZZ2140
+002506	243150		dac ZZ3140
-	scale ~cs, 8s, ~scn
+002507	203147	    	lac ZZ1141
+002510	675377		sar ZZ2141
+002511	243114		dac ZZ3141
 002512	203150		lac ~ssn
 002513	243155		dac ~ssm
 002514	403114		add ~scn
 002515	243156		dac ~ssc
 002516	243157		dac ~ssd
 002517	203150		lac ~ssn
 002520	423114		sub ~scn
 002521	243160		dac ~csn
 002522	761000		cma
 002523	243161		dac ~csm
 002524	203114		lac ~scn
 002525	243162		dac ~scm
 002526	764200		cla cli-opr
 002527	724007		dpy-4000
 002530		sp5,
 002530	602530		jmp .
 002531		sq6,
 002531	730000		ioh
-	ranct sar 9s, sar 4s, ~src
-    	random
+002532	200031	    	lac ran
+002533	671001		rar 1s
+002534	063041		xor (355760
+002535	403042		add (355670
+002536	240031		dac ran
+002537	675777		ZZ1142
+002540	675017		ZZ2142
+002541	640400		sma
+002542	761000		cma
+002543	243163		dac ZZ3142
 002544	223143		lio ~scw
 002545	662003		ril 2s
 002546	652000		spi i				/ not blasting
 002547	602574		jmp sq9				/ no tail
 002550		sq7,
-	scale ~sn, 8s, ~ssn
+002550	203144	    	lac ZZ1144
+002551	675377		sar ZZ2144
+002552	243150		dac ZZ3144
-	scale ~cs, 8s, ~scn
+002553	203147	    	lac ZZ1145
+002554	675377		sar ZZ2145
+002555	243114		dac ZZ3145
-	count i ~mfu, st2
+002556	473123	    	isp ZZ1146
+002557	602562		jmp ZZ2146
 002560	353123		dzm i ~mfu
 002561	602574		jmp sq9
 002562		st2,
-	yincr ~sx1, ~sy1, sub
+002562	203153	    	lac ZZ2147
+002563	423114		ZZ3147 ~scn
+002564	243153		dac ZZ2147
+002565	203151		lac ZZ1147
+002566	403150		-ZZ3147+add+sub ~ssn
+002567	243151		dac ZZ1147
-	dispt i, ~sy1
+002570	000000		ZZ3148=ZZ3148+ZZ3148
+002570	000000		ZZ3148=ZZ3148+ZZ3148
+002570	000000		ZZ3148=ZZ3148+ZZ3148
+002570	000000		ZZ3148=ZZ3148+ZZ3148
+002570	000000		ZZ3148=ZZ3148+ZZ3148
+002570	000000		ZZ3148=ZZ3148+ZZ3148
+002570	223153		lio ZZ2148
+002571	720007		dpy-ZZ1148+ZZ3148
-	count ~src,sq7
+002572	463163	    	isp ZZ1149
+002573	602550		jmp ZZ2149
 002574		sq9,
-	count i ma1, sr5		/ check if torp tube reloaded
+002574	471772	    	isp ZZ1150
+002575	602667		jmp ZZ2150
 002576	351772		dzm i ma1			/ prevent count around
 002577		mco,
 002577	202577		lac .				/ previous control word
 002600	761000		cma
 002601	650030		szs i 30
 002602	761200		clc
 002603	023143		and ~scw			/ present control word
 002604	661007		ral 3s				/ torpedo bit to bit 0
 002605	640400		sma
 002606	602667		jmp sr5				/ no launch
-	count i ~mtr, st1		/ check if torpedos exhausted
+002607	473124	    	isp ZZ1151
+002610	602613		jmp ZZ2151
 002611	353124		dzm i ~mtr			/ prevent count around
 002612	602667		jmp sr5
 002613		st1,
-	init sr1, mtb			/ search for unused object
+002613	703365	    	law ZZ2152
+002614	262615		dap ZZ1152
 002615		sr1,
 002615	202615		lac .
 002616	650100		sza i				/ 0 if unused
 002617	602625		jmp sr2
-	index sr1, (lac mtb+nob, sr1
+002620	442615	    	idx ZZ1153
+002621	523105		sas ZZ2153
+002622	602615		jmp ZZ3153
 002623	760400		hlt				/ no space for new objects
 002624	602623		jmp .-1
-////
 002625		sr2,
 002625	203106		lac (tcr
 002626	252615		dac i sr1
 002627	700030		law nob
 002630	402615		add sr1
 002631	262633		dap ss3
 002632	223152		lio ~stx
 002633		ss3,
 002633	322633		dio .
 002634	403066		add (nob
 002635	262637		dap ss4
 002636	223154		lio ~sty
 002637		ss4,
 002637	322637		dio .
 002640	403066		add (nob
 002641	262664		dap sr6
 002642	403066		add (nob
 002643	262666		dap sr7
 002644	403066		add (nob
 002645	262654		dap sr3
 002646	403066		add (nob
 002647	262660		dap sr4
 002650	203144		lac ~sn
 002651	100007		xct tvl
 002652	761000		cma
 002653	413121		add i ~mdx
 002654		sr3,
 002654	242654		dac .
 002655	203147		lac ~cs
 002656	100007		xct tvl
 002657	413122		add i ~mdy
 002660		sr4,
 002660	242660		dac .
 002661	100010		xct rlt
 002662	251772		dac i ma1			/ permit torp tubes to cool
 002663		trp,
 002663	100011		xct tlf				/ life of torpedo
 002664		sr6,
 002664	242664		dac .
 002665	700020		law 20
 002666		sr7,
 002666	262666		dap .				/ length of torp calc
 002667		sr5,
-	count i ~mh3, st3		/ hyperbutton active?
+002667	473127	    	isp ZZ1154
+002670	602713		jmp ZZ2154
 002671	353127		dzm i ~mh3
 002672	213126		lac i ~mh2
 002673	650100		sza i
 002674	602713		jmp st3
 002675	203143		lac ~scw
 002676	761000		cma
 002677	052577		ior i mco
 002700	023107		and (600000
 002701	640100		sza
 002702	602713		jmp st3
 002703	211703		lac i ml1
 002704	253125		dac i ~mh1
 002705	203110		lac (hp1 400000
 002706	251703		dac i ml1
 002707	100023		xct hd1
 002710	251772		dac i ma1
 002711	700003		law 3
 002712	252006		dac i mb1
 002713		st3,
 002713		srt,
 002713	602713		jmp .
-////
-/ here to handle spaceships into star
-/ spaceship in star
 002714		poh,
 002714	353121		dzm i ~mdx
 002715	353122		dzm i ~mdy
 002716	640050		szs 50
 002717	602730		jmp po1
 002720	202767		lac (377777
 002721	251737		dac i mx1
 002722	251747		dac i my1
 002723	212006		lac i mb1
 002724	243150		dac ~ssn
-	count ~ssn, .
+002725	463150	    	isp ZZ1155
+002726	602725		jmp ZZ2155
 002727	602713		jmp srt
 002730		po1,
 002730	203103		lac (mex 400000	/ now go bang
 002731	251703		dac i ml1
 002732	710010		law i 10
 002733	251772		dac i ma1
 002734	602713		jmp srt
-////
-/ outlines of spaceships
 002735		ot1,
 002735	111131		111131
 002736	111111		111111
 002737	111111		111111
 002740	111163		111163
 002741	311111		311111
 002742	146111		146111
 002743	111114		111114
 002744	700000		700000
 002745	000005	. 5/
 002746		ot2,
 002746	013113		013113
 002747	113111		113111
 002750	116313		116313
 002751	131111		131111
 002752	161151		161151
 002753	111633		111633
 002754	365114		365114
 002755	700000		700000
 002756	000005	. 5/
 002757	203164		lac ~ssa	/ To fix assembler bug - ~ssa only referenced in lit
 002760			constants
+002760	062210	62210
+002761	311040	311040
+002762	242763	242763
+002763	756103	756103
+002764	121312	121312
+002765	532511	532511
+002766	144417	144417
+002767	377777	377777
+002770	000001	1
+002771	760015	stf 5
+002772	203151	lac ~sx1
+002773	223153	lio ~sy1
+002774	663777	rcl 9s
+002775	000444	a11
+002776	640005	szf 5
+002777	000004	4
+003000	243151	dac ~sx1
+003001	323153	dio ~sy1
+003002	602531	jmp sq6
+003003	760005	clf 5
+003004	203162	lac ~scm
+003005	761000	cma
+003006	243162	dac ~scm
+003007	203155	lac ~ssm
+003010	243155	dac ~ssm
+003011	203161	lac ~csm
+003012	223157	lio ~ssd
+003013	243157	dac ~ssd
+003014	323161	dio ~csm
+003015	203156	lac ~ssc
+003016	223160	lio ~csn
+003017	243160	dac ~csn
+003020	323156	dio ~ssc
+003021	403150	add ~ssn
+003022	423114	sub ~scn
+003023	730000	ioh
+003024	724007	dpy-4000
+003025	403162	add ~scm
+003026	403155	add ~ssm
+003027	403156	add ~ssc
+003030	423161	sub ~csm
+003031	423162	sub ~scm
+003032	423155	sub ~ssm
+003033	403160	add ~csn
+003034	423157	sub ~ssd
+003035	243164	dac ~ssa
+003036	323115	dio ~ssi
+003037	203164	lac ~ssa
+003040	223115	lio ~ssi
+003041	355760	355760
+003042	355670	355670
+003043	400340	add 340
+003044	000716	bds
+003045	002000	2000
+003046	001000	1000
+003047	226022	lio ZZ298+2
+003050	761777	2000 -20000
+003051	006022	ZZ298+2
+003052	226000	lio ZZ198
+003053	226044	lio ZZ299+2
+003054	006044	ZZ299+2
+003055	226022	lio ZZ199
+003056	226306	lio ZZ2100+2
+003057	006306	ZZ2100+2
+003060	226044	lio ZZ1100
+003061	227652	lio ZZ2101+2
+003062	007652	ZZ2101+2
+003063	226306	lio ZZ1101
+003064	020000	20000
+003065	773777	ZZ2102
+003066	000030	nob
+003067	000002	2
+003070	000040	40
+003071	000037	37
+003072	343661	dzm ZZ2106+1
+003073	200000	200000
+003074	144420	144420
+003075	203415	lac mtb nob
+003076	203414	lac mtb nob-1
+003077	675007	sar 3s
+003100	000140	140
+003101	000777	777
+003102	667000	scl
+003103	402052	mex 400000
+003104	400000	400000
+003105	203415	lac mtb+nob
+003106	002136	tcr
+003107	600000	600000
+003110	402170	hp1 400000
 003111	000000		0
 003112			variables
+003112	000000	occ
+003113	000000	oci
+003114	000000	scn
+003115	000000	ssi
+003116	000000	bx
+003117	000000	by
+003120	000000	mtc
+003121	000000	mdx
+003122	000000	mdy
+003123	000000	mfu
+003124	000000	mtr
+003125	000000	mh1
+003126	000000	mh2
+003127	000000	mh3
+003130	000000	mh4
+003131	000000	ntd
+003132	000000	1sc
+003133	000000	2sc
+003134	000000	cwg
+003135	000000	gct
+003136	000000	moc
+003137	000000	mt1
+003140	000000	mas
+003141	000000	mxc
+003142	000000	hpt
+003143	000000	scw
+003144	000000	sn
+003145	000000	t1
+003146	000000	t2
+003147	000000	cs
+003150	000000	ssn
+003151	000000	sx1
+003152	000000	stx
+003153	000000	sy1
+003154	000000	sty
+003155	000000	ssm
+003156	000000	ssc
+003157	000000	ssd
+003160	000000	csn
+003161	000000	csm
+003162	000000	scm
+003163	000000	src
+003164	000000	ssa
 003165		p,
 003365			. 200/		/ space for patches
 003365		mtb,
-				/ table of objects and their properties
 006000			6000/
-/stars 1 3/13/62 prs.
 006000			decimal
-	define mark X, Y
-	repeat 10, Y=Y+Y
-	0 8192 -X
-	0 Y
-	terminate
 006000		1j,
-	 mark 1537, 371		/87 taur, aldebaran
+006000	001346		ZZ2156=ZZ2156+ZZ2156
+006000	002714		ZZ2156=ZZ2156+ZZ2156
+006000	005630		ZZ2156=ZZ2156+ZZ2156
+006000	013460		ZZ2156=ZZ2156+ZZ2156
+006000	027140		ZZ2156=ZZ2156+ZZ2156
+006000	056300		ZZ2156=ZZ2156+ZZ2156
+006000	134600		ZZ2156=ZZ2156+ZZ2156
+006000	271400		ZZ2156=ZZ2156+ZZ2156
+006000	014777		0 8192 -ZZ1156
+006001	271400		0 ZZ2156
-	 mark 1762, -189	/19 orio, rigel
+006002	777204		ZZ2157=ZZ2157+ZZ2157
+006002	776410		ZZ2157=ZZ2157+ZZ2157
+006002	775020		ZZ2157=ZZ2157+ZZ2157
+006002	772040		ZZ2157=ZZ2157+ZZ2157
+006002	764100		ZZ2157=ZZ2157+ZZ2157
+006002	750200		ZZ2157=ZZ2157+ZZ2157
+006002	720400		ZZ2157=ZZ2157+ZZ2157
+006002	641000		ZZ2157=ZZ2157+ZZ2157
+006002	014436		0 8192 -ZZ1157
+006003	641000		0 ZZ2157
-	 mark 1990, 168		/58 orio, betelgeuze
+006004	000520		ZZ2158=ZZ2158+ZZ2158
+006004	001240		ZZ2158=ZZ2158+ZZ2158
+006004	002500		ZZ2158=ZZ2158+ZZ2158
+006004	005200		ZZ2158=ZZ2158+ZZ2158
+006004	012400		ZZ2158=ZZ2158+ZZ2158
+006004	025000		ZZ2158=ZZ2158+ZZ2158
+006004	052000		ZZ2158=ZZ2158+ZZ2158
+006004	124000		ZZ2158=ZZ2158+ZZ2158
+006004	014072		0 8192 -ZZ1158
+006005	124000		0 ZZ2158
-	 mark 2280, -377	/9 cmaj, sirius
+006006	776414		ZZ2159=ZZ2159+ZZ2159
+006006	775030		ZZ2159=ZZ2159+ZZ2159
+006006	772060		ZZ2159=ZZ2159+ZZ2159
+006006	764140		ZZ2159=ZZ2159+ZZ2159
+006006	750300		ZZ2159=ZZ2159+ZZ2159
+006006	720600		ZZ2159=ZZ2159+ZZ2159
+006006	641400		ZZ2159=ZZ2159+ZZ2159
+006006	503000		ZZ2159=ZZ2159+ZZ2159
+006006	013430		0 8192 -ZZ1159
+006007	503000		0 ZZ2159
-	 mark 2583, 125		/25 cmin, procyon
+006010	000372		ZZ2160=ZZ2160+ZZ2160
+006010	000764		ZZ2160=ZZ2160+ZZ2160
+006010	001750		ZZ2160=ZZ2160+ZZ2160
+006010	003720		ZZ2160=ZZ2160+ZZ2160
+006010	007640		ZZ2160=ZZ2160+ZZ2160
+006010	017500		ZZ2160=ZZ2160+ZZ2160
+006010	037200		ZZ2160=ZZ2160+ZZ2160
+006010	076400		ZZ2160=ZZ2160+ZZ2160
+006010	012751		0 8192 -ZZ1160
+006011	076400		0 ZZ2160
-	 mark 3431, 283		/32 leon, regulus
+006012	001066		ZZ2161=ZZ2161+ZZ2161
+006012	002154		ZZ2161=ZZ2161+ZZ2161
+006012	004330		ZZ2161=ZZ2161+ZZ2161
+006012	010660		ZZ2161=ZZ2161+ZZ2161
+006012	021540		ZZ2161=ZZ2161+ZZ2161
+006012	043300		ZZ2161=ZZ2161+ZZ2161
+006012	106600		ZZ2161=ZZ2161+ZZ2161
+006012	215400		ZZ2161=ZZ2161+ZZ2161
+006012	011231		0 8192 -ZZ1161
+006013	215400		0 ZZ2161
-	 mark 4551, -242	/67 virg, spica
+006014	777032		ZZ2162=ZZ2162+ZZ2162
+006014	776064		ZZ2162=ZZ2162+ZZ2162
+006014	774150		ZZ2162=ZZ2162+ZZ2162
+006014	770320		ZZ2162=ZZ2162+ZZ2162
+006014	760640		ZZ2162=ZZ2162+ZZ2162
+006014	741500		ZZ2162=ZZ2162+ZZ2162
+006014	703200		ZZ2162=ZZ2162+ZZ2162
+006014	606400		ZZ2162=ZZ2162+ZZ2162
+006014	007071		0 8192 -ZZ1162
+006015	606400		0 ZZ2162
-	 mark 4842, 448		/16 boot, arcturus
+006016	001600		ZZ2163=ZZ2163+ZZ2163
+006016	003400		ZZ2163=ZZ2163+ZZ2163
+006016	007000		ZZ2163=ZZ2163+ZZ2163
+006016	016000		ZZ2163=ZZ2163+ZZ2163
+006016	034000		ZZ2163=ZZ2163+ZZ2163
+006016	070000		ZZ2163=ZZ2163+ZZ2163
+006016	160000		ZZ2163=ZZ2163+ZZ2163
+006016	340000		ZZ2163=ZZ2163+ZZ2163
+006016	006426		0 8192 -ZZ1163
+006017	340000		0 ZZ2163
 006020		1q,
-	 mark 6747, 196		/53 aqil, altair
+006020	000610		ZZ2164=ZZ2164+ZZ2164
+006020	001420		ZZ2164=ZZ2164+ZZ2164
+006020	003040		ZZ2164=ZZ2164+ZZ2164
+006020	006100		ZZ2164=ZZ2164+ZZ2164
+006020	014200		ZZ2164=ZZ2164+ZZ2164
+006020	030400		ZZ2164=ZZ2164+ZZ2164
+006020	061000		ZZ2164=ZZ2164+ZZ2164
+006020	142000		ZZ2164=ZZ2164+ZZ2164
+006020	002645		0 8192 -ZZ1164
+006021	142000		0 ZZ2164
 006022		2j,
-	 mark 1819, 143		/24 orio, bellatrix
+006022	000436		ZZ2165=ZZ2165+ZZ2165
+006022	001074		ZZ2165=ZZ2165+ZZ2165
+006022	002170		ZZ2165=ZZ2165+ZZ2165
+006022	004360		ZZ2165=ZZ2165+ZZ2165
+006022	010740		ZZ2165=ZZ2165+ZZ2165
+006022	021700		ZZ2165=ZZ2165+ZZ2165
+006022	043600		ZZ2165=ZZ2165+ZZ2165
+006022	107400		ZZ2165=ZZ2165+ZZ2165
+006022	014345		0 8192 -ZZ1165
+006023	107400		0 ZZ2165
-	 mark 1884, -29		/46 orio
+006024	777704		ZZ2166=ZZ2166+ZZ2166
+006024	777610		ZZ2166=ZZ2166+ZZ2166
+006024	777420		ZZ2166=ZZ2166+ZZ2166
+006024	777040		ZZ2166=ZZ2166+ZZ2166
+006024	776100		ZZ2166=ZZ2166+ZZ2166
+006024	774200		ZZ2166=ZZ2166+ZZ2166
+006024	770400		ZZ2166=ZZ2166+ZZ2166
+006024	761000		ZZ2166=ZZ2166+ZZ2166
+006024	014244		0 8192 -ZZ1166
+006025	761000		0 ZZ2166
-	 mark 1910, -46		/50 orio
+006026	777642		ZZ2167=ZZ2167+ZZ2167
+006026	777504		ZZ2167=ZZ2167+ZZ2167
+006026	777210		ZZ2167=ZZ2167+ZZ2167
+006026	776420		ZZ2167=ZZ2167+ZZ2167
+006026	775040		ZZ2167=ZZ2167+ZZ2167
+006026	772100		ZZ2167=ZZ2167+ZZ2167
+006026	764200		ZZ2167=ZZ2167+ZZ2167
+006026	750400		ZZ2167=ZZ2167+ZZ2167
+006026	014212		0 8192 -ZZ1167
+006027	750400		0 ZZ2167
-	 mark 1951, -221	/53 orio
+006030	777104		ZZ2168=ZZ2168+ZZ2168
+006030	776210		ZZ2168=ZZ2168+ZZ2168
+006030	774420		ZZ2168=ZZ2168+ZZ2168
+006030	771040		ZZ2168=ZZ2168+ZZ2168
+006030	762100		ZZ2168=ZZ2168+ZZ2168
+006030	744200		ZZ2168=ZZ2168+ZZ2168
+006030	710400		ZZ2168=ZZ2168+ZZ2168
+006030	621000		ZZ2168=ZZ2168+ZZ2168
+006030	014141		0 8192 -ZZ1168
+006031	621000		0 ZZ2168
-	 mark 2152, -407	/ 2 cmaj
+006032	776320		ZZ2169=ZZ2169+ZZ2169
+006032	774640		ZZ2169=ZZ2169+ZZ2169
+006032	771500		ZZ2169=ZZ2169+ZZ2169
+006032	763200		ZZ2169=ZZ2169+ZZ2169
+006032	746400		ZZ2169=ZZ2169+ZZ2169
+006032	715000		ZZ2169=ZZ2169+ZZ2169
+006032	632000		ZZ2169=ZZ2169+ZZ2169
+006032	464000		ZZ2169=ZZ2169+ZZ2169
+006032	013630		0 8192 -ZZ1169
+006033	464000		0 ZZ2169
-	 mark 2230, 375		/24 gemi
+006034	001356		ZZ2170=ZZ2170+ZZ2170
+006034	002734		ZZ2170=ZZ2170+ZZ2170
+006034	005670		ZZ2170=ZZ2170+ZZ2170
+006034	013560		ZZ2170=ZZ2170+ZZ2170
+006034	027340		ZZ2170=ZZ2170+ZZ2170
+006034	056700		ZZ2170=ZZ2170+ZZ2170
+006034	135600		ZZ2170=ZZ2170+ZZ2170
+006034	273400		ZZ2170=ZZ2170+ZZ2170
+006034	013512		0 8192 -ZZ1170
+006035	273400		0 ZZ2170
-	 mark 3201, -187	/30 hyda, alphard
+006036	777210		ZZ2171=ZZ2171+ZZ2171
+006036	776420		ZZ2171=ZZ2171+ZZ2171
+006036	775040		ZZ2171=ZZ2171+ZZ2171
+006036	772100		ZZ2171=ZZ2171+ZZ2171
+006036	764200		ZZ2171=ZZ2171+ZZ2171
+006036	750400		ZZ2171=ZZ2171+ZZ2171
+006036	721000		ZZ2171=ZZ2171+ZZ2171
+006036	642000		ZZ2171=ZZ2171+ZZ2171
+006036	011577		0 8192 -ZZ1171
+006037	642000		0 ZZ2171
-	 mark 4005, 344		/94 leon, denebola
+006040	001260		ZZ2172=ZZ2172+ZZ2172
+006040	002540		ZZ2172=ZZ2172+ZZ2172
+006040	005300		ZZ2172=ZZ2172+ZZ2172
+006040	012600		ZZ2172=ZZ2172+ZZ2172
+006040	025400		ZZ2172=ZZ2172+ZZ2172
+006040	053000		ZZ2172=ZZ2172+ZZ2172
+006040	126000		ZZ2172=ZZ2172+ZZ2172
+006040	254000		ZZ2172=ZZ2172+ZZ2172
+006040	010133		0 8192 -ZZ1172
+006041	254000		0 ZZ2172
 006042		2q,
-	 mark 5975, 288		/55 ophi
+006042	001100		ZZ2173=ZZ2173+ZZ2173
+006042	002200		ZZ2173=ZZ2173+ZZ2173
+006042	004400		ZZ2173=ZZ2173+ZZ2173
+006042	011000		ZZ2173=ZZ2173+ZZ2173
+006042	022000		ZZ2173=ZZ2173+ZZ2173
+006042	044000		ZZ2173=ZZ2173+ZZ2173
+006042	110000		ZZ2173=ZZ2173+ZZ2173
+006042	220000		ZZ2173=ZZ2173+ZZ2173
+006042	004251		0 8192 -ZZ1173
+006043	220000		0 ZZ2173
 006044		3j,
-	 mark   46, 333		/88 pegs, algenib
+006044	001232		ZZ2174=ZZ2174+ZZ2174
+006044	002464		ZZ2174=ZZ2174+ZZ2174
+006044	005150		ZZ2174=ZZ2174+ZZ2174
+006044	012320		ZZ2174=ZZ2174+ZZ2174
+006044	024640		ZZ2174=ZZ2174+ZZ2174
+006044	051500		ZZ2174=ZZ2174+ZZ2174
+006044	123200		ZZ2174=ZZ2174+ZZ2174
+006044	246400		ZZ2174=ZZ2174+ZZ2174
+006044	017722		0 8192 -ZZ1174
+006045	246400		0 ZZ2174
-	 mark  362, -244  	/31 ceti
+006046	777026		ZZ2175=ZZ2175+ZZ2175
+006046	776054		ZZ2175=ZZ2175+ZZ2175
+006046	774130		ZZ2175=ZZ2175+ZZ2175
+006046	770260		ZZ2175=ZZ2175+ZZ2175
+006046	760540		ZZ2175=ZZ2175+ZZ2175
+006046	741300		ZZ2175=ZZ2175+ZZ2175
+006046	702600		ZZ2175=ZZ2175+ZZ2175
+006046	605400		ZZ2175=ZZ2175+ZZ2175
+006046	017226		0 8192 -ZZ1175
+006047	605400		0 ZZ2175
-	 mark  490, 338		/99 pisc
+006050	001244		ZZ2176=ZZ2176+ZZ2176
+006050	002510		ZZ2176=ZZ2176+ZZ2176
+006050	005220		ZZ2176=ZZ2176+ZZ2176
+006050	012440		ZZ2176=ZZ2176+ZZ2176
+006050	025100		ZZ2176=ZZ2176+ZZ2176
+006050	052200		ZZ2176=ZZ2176+ZZ2176
+006050	124400		ZZ2176=ZZ2176+ZZ2176
+006050	251000		ZZ2176=ZZ2176+ZZ2176
+006050	017026		0 8192 -ZZ1176
+006051	251000		0 ZZ2176
-	 mark  566, -375 	/52 ceti
+006052	776420		ZZ2177=ZZ2177+ZZ2177
+006052	775040		ZZ2177=ZZ2177+ZZ2177
+006052	772100		ZZ2177=ZZ2177+ZZ2177
+006052	764200		ZZ2177=ZZ2177+ZZ2177
+006052	750400		ZZ2177=ZZ2177+ZZ2177
+006052	721000		ZZ2177=ZZ2177+ZZ2177
+006052	642000		ZZ2177=ZZ2177+ZZ2177
+006052	504000		ZZ2177=ZZ2177+ZZ2177
+006052	016712		0 8192 -ZZ1177
+006053	504000		0 ZZ2177
-	 mark  621, 462		/ 6 arie
+006054	001634		ZZ2178=ZZ2178+ZZ2178
+006054	003470		ZZ2178=ZZ2178+ZZ2178
+006054	007160		ZZ2178=ZZ2178+ZZ2178
+006054	016340		ZZ2178=ZZ2178+ZZ2178
+006054	034700		ZZ2178=ZZ2178+ZZ2178
+006054	071600		ZZ2178=ZZ2178+ZZ2178
+006054	163400		ZZ2178=ZZ2178+ZZ2178
+006054	347000		ZZ2178=ZZ2178+ZZ2178
+006054	016623		0 8192 -ZZ1178
+006055	347000		0 ZZ2178
-	 mark 764, -78		/68 ceti, mira
+006056	777542		ZZ2179=ZZ2179+ZZ2179
+006056	777304		ZZ2179=ZZ2179+ZZ2179
+006056	776610		ZZ2179=ZZ2179+ZZ2179
+006056	775420		ZZ2179=ZZ2179+ZZ2179
+006056	773040		ZZ2179=ZZ2179+ZZ2179
+006056	766100		ZZ2179=ZZ2179+ZZ2179
+006056	754200		ZZ2179=ZZ2179+ZZ2179
+006056	730400		ZZ2179=ZZ2179+ZZ2179
+006056	016404		0 8192 -ZZ1179
+006057	730400		0 ZZ2179
-	 mark  900, 64		/86 ceti
+006060	000200		ZZ2180=ZZ2180+ZZ2180
+006060	000400		ZZ2180=ZZ2180+ZZ2180
+006060	001000		ZZ2180=ZZ2180+ZZ2180
+006060	002000		ZZ2180=ZZ2180+ZZ2180
+006060	004000		ZZ2180=ZZ2180+ZZ2180
+006060	010000		ZZ2180=ZZ2180+ZZ2180
+006060	020000		ZZ2180=ZZ2180+ZZ2180
+006060	040000		ZZ2180=ZZ2180+ZZ2180
+006060	016174		0 8192 -ZZ1180
+006061	040000		0 ZZ2180
-	 mark 1007, 84		/92 ceti
+006062	000250		ZZ2181=ZZ2181+ZZ2181
+006062	000520		ZZ2181=ZZ2181+ZZ2181
+006062	001240		ZZ2181=ZZ2181+ZZ2181
+006062	002500		ZZ2181=ZZ2181+ZZ2181
+006062	005200		ZZ2181=ZZ2181+ZZ2181
+006062	012400		ZZ2181=ZZ2181+ZZ2181
+006062	025000		ZZ2181=ZZ2181+ZZ2181
+006062	052000		ZZ2181=ZZ2181+ZZ2181
+006062	016021		0 8192 -ZZ1181
+006063	052000		0 ZZ2181
-	 mark 1243, -230	/23 erid
+006064	777062		ZZ2182=ZZ2182+ZZ2182
+006064	776144		ZZ2182=ZZ2182+ZZ2182
+006064	774310		ZZ2182=ZZ2182+ZZ2182
+006064	770620		ZZ2182=ZZ2182+ZZ2182
+006064	761440		ZZ2182=ZZ2182+ZZ2182
+006064	743100		ZZ2182=ZZ2182+ZZ2182
+006064	706200		ZZ2182=ZZ2182+ZZ2182
+006064	614400		ZZ2182=ZZ2182+ZZ2182
+006064	015445		0 8192 -ZZ1182
+006065	614400		0 ZZ2182
-	 mark 1328, -314	/34 erid
+006066	776612		ZZ2183=ZZ2183+ZZ2183
+006066	775424		ZZ2183=ZZ2183+ZZ2183
+006066	773050		ZZ2183=ZZ2183+ZZ2183
+006066	766120		ZZ2183=ZZ2183+ZZ2183
+006066	754240		ZZ2183=ZZ2183+ZZ2183
+006066	730500		ZZ2183=ZZ2183+ZZ2183
+006066	661200		ZZ2183=ZZ2183+ZZ2183
+006066	542400		ZZ2183=ZZ2183+ZZ2183
+006066	015320		0 8192 -ZZ1183
+006067	542400		0 ZZ2183
-	 mark 1495, 432		/74 taur
+006070	001540		ZZ2184=ZZ2184+ZZ2184
+006070	003300		ZZ2184=ZZ2184+ZZ2184
+006070	006600		ZZ2184=ZZ2184+ZZ2184
+006070	015400		ZZ2184=ZZ2184+ZZ2184
+006070	033000		ZZ2184=ZZ2184+ZZ2184
+006070	066000		ZZ2184=ZZ2184+ZZ2184
+006070	154000		ZZ2184=ZZ2184+ZZ2184
+006070	330000		ZZ2184=ZZ2184+ZZ2184
+006070	015051		0 8192 -ZZ1184
+006071	330000		0 ZZ2184
-	 mark 1496, 356		/78 taur
+006072	001310		ZZ2185=ZZ2185+ZZ2185
+006072	002620		ZZ2185=ZZ2185+ZZ2185
+006072	005440		ZZ2185=ZZ2185+ZZ2185
+006072	013100		ZZ2185=ZZ2185+ZZ2185
+006072	026200		ZZ2185=ZZ2185+ZZ2185
+006072	054400		ZZ2185=ZZ2185+ZZ2185
+006072	131000		ZZ2185=ZZ2185+ZZ2185
+006072	262000		ZZ2185=ZZ2185+ZZ2185
+006072	015050		0 8192 -ZZ1185
+006073	262000		0 ZZ2185
-	 mark 1618, 154		/ 1 orio
+006074	000464		ZZ2186=ZZ2186+ZZ2186
+006074	001150		ZZ2186=ZZ2186+ZZ2186
+006074	002320		ZZ2186=ZZ2186+ZZ2186
+006074	004640		ZZ2186=ZZ2186+ZZ2186
+006074	011500		ZZ2186=ZZ2186+ZZ2186
+006074	023200		ZZ2186=ZZ2186+ZZ2186
+006074	046400		ZZ2186=ZZ2186+ZZ2186
+006074	115000		ZZ2186=ZZ2186+ZZ2186
+006074	014656		0 8192 -ZZ1186
+006075	115000		0 ZZ2186
-	 mark 1644, 52		/ 8 orio
+006076	000150		ZZ2187=ZZ2187+ZZ2187
+006076	000320		ZZ2187=ZZ2187+ZZ2187
+006076	000640		ZZ2187=ZZ2187+ZZ2187
+006076	001500		ZZ2187=ZZ2187+ZZ2187
+006076	003200		ZZ2187=ZZ2187+ZZ2187
+006076	006400		ZZ2187=ZZ2187+ZZ2187
+006076	015000		ZZ2187=ZZ2187+ZZ2187
+006076	032000		ZZ2187=ZZ2187+ZZ2187
+006076	014624		0 8192 -ZZ1187
+006077	032000		0 ZZ2187
-	 mark 1723, -119	/67 erid
+006100	777420		ZZ2188=ZZ2188+ZZ2188
+006100	777040		ZZ2188=ZZ2188+ZZ2188
+006100	776100		ZZ2188=ZZ2188+ZZ2188
+006100	774200		ZZ2188=ZZ2188+ZZ2188
+006100	770400		ZZ2188=ZZ2188+ZZ2188
+006100	761000		ZZ2188=ZZ2188+ZZ2188
+006100	742000		ZZ2188=ZZ2188+ZZ2188
+006100	704000		ZZ2188=ZZ2188+ZZ2188
+006100	014505		0 8192 -ZZ1188
+006101	704000		0 ZZ2188
-	 mark 1755, -371	/ 5 leps
+006102	776430		ZZ2189=ZZ2189+ZZ2189
+006102	775060		ZZ2189=ZZ2189+ZZ2189
+006102	772140		ZZ2189=ZZ2189+ZZ2189
+006102	764300		ZZ2189=ZZ2189+ZZ2189
+006102	750600		ZZ2189=ZZ2189+ZZ2189
+006102	721400		ZZ2189=ZZ2189+ZZ2189
+006102	643000		ZZ2189=ZZ2189+ZZ2189
+006102	506000		ZZ2189=ZZ2189+ZZ2189
+006102	014445		0 8192 -ZZ1189
+006103	506000		0 ZZ2189
-	 mark 1779, -158	/20 orio
+006104	777302		ZZ2190=ZZ2190+ZZ2190
+006104	776604		ZZ2190=ZZ2190+ZZ2190
+006104	775410		ZZ2190=ZZ2190+ZZ2190
+006104	773020		ZZ2190=ZZ2190+ZZ2190
+006104	766040		ZZ2190=ZZ2190+ZZ2190
+006104	754100		ZZ2190=ZZ2190+ZZ2190
+006104	730200		ZZ2190=ZZ2190+ZZ2190
+006104	660400		ZZ2190=ZZ2190+ZZ2190
+006104	014415		0 8192 -ZZ1190
+006105	660400		0 ZZ2190
-	 mark 1817, -57		/28 orio
+006106	777614		ZZ2191=ZZ2191+ZZ2191
+006106	777430		ZZ2191=ZZ2191+ZZ2191
+006106	777060		ZZ2191=ZZ2191+ZZ2191
+006106	776140		ZZ2191=ZZ2191+ZZ2191
+006106	774300		ZZ2191=ZZ2191+ZZ2191
+006106	770600		ZZ2191=ZZ2191+ZZ2191
+006106	761400		ZZ2191=ZZ2191+ZZ2191
+006106	743000		ZZ2191=ZZ2191+ZZ2191
+006106	014347		0 8192 -ZZ1191
+006107	743000		0 ZZ2191
-	 mark 1843, -474	/ 9 leps
+006110	776112		ZZ2192=ZZ2192+ZZ2192
+006110	774224		ZZ2192=ZZ2192+ZZ2192
+006110	770450		ZZ2192=ZZ2192+ZZ2192
+006110	761120		ZZ2192=ZZ2192+ZZ2192
+006110	742240		ZZ2192=ZZ2192+ZZ2192
+006110	704500		ZZ2192=ZZ2192+ZZ2192
+006110	611200		ZZ2192=ZZ2192+ZZ2192
+006110	422400		ZZ2192=ZZ2192+ZZ2192
+006110	014315		0 8192 -ZZ1192
+006111	422400		0 ZZ2192
-	 mark 1860, -8		/34 orio
+006112	777756		ZZ2193=ZZ2193+ZZ2193
+006112	777734		ZZ2193=ZZ2193+ZZ2193
+006112	777670		ZZ2193=ZZ2193+ZZ2193
+006112	777560		ZZ2193=ZZ2193+ZZ2193
+006112	777340		ZZ2193=ZZ2193+ZZ2193
+006112	776700		ZZ2193=ZZ2193+ZZ2193
+006112	775600		ZZ2193=ZZ2193+ZZ2193
+006112	773400		ZZ2193=ZZ2193+ZZ2193
+006112	014274		0 8192 -ZZ1193
+006113	773400		0 ZZ2193
-	 mark 1868, -407	/11 leps
+006114	776320		ZZ2194=ZZ2194+ZZ2194
+006114	774640		ZZ2194=ZZ2194+ZZ2194
+006114	771500		ZZ2194=ZZ2194+ZZ2194
+006114	763200		ZZ2194=ZZ2194+ZZ2194
+006114	746400		ZZ2194=ZZ2194+ZZ2194
+006114	715000		ZZ2194=ZZ2194+ZZ2194
+006114	632000		ZZ2194=ZZ2194+ZZ2194
+006114	464000		ZZ2194=ZZ2194+ZZ2194
+006114	014264		0 8192 -ZZ1194
+006115	464000		0 ZZ2194
-	 mark 1875, 225		/39 orio
+006116	000702		ZZ2195=ZZ2195+ZZ2195
+006116	001604		ZZ2195=ZZ2195+ZZ2195
+006116	003410		ZZ2195=ZZ2195+ZZ2195
+006116	007020		ZZ2195=ZZ2195+ZZ2195
+006116	016040		ZZ2195=ZZ2195+ZZ2195
+006116	034100		ZZ2195=ZZ2195+ZZ2195
+006116	070200		ZZ2195=ZZ2195+ZZ2195
+006116	160400		ZZ2195=ZZ2195+ZZ2195
+006116	014255		0 8192 -ZZ1195
+006117	160400		0 ZZ2195
-	 mark 1880, -136	/44 orio
+006120	777356		ZZ2196=ZZ2196+ZZ2196
+006120	776734		ZZ2196=ZZ2196+ZZ2196
+006120	775670		ZZ2196=ZZ2196+ZZ2196
+006120	773560		ZZ2196=ZZ2196+ZZ2196
+006120	767340		ZZ2196=ZZ2196+ZZ2196
+006120	756700		ZZ2196=ZZ2196+ZZ2196
+006120	735600		ZZ2196=ZZ2196+ZZ2196
+006120	673400		ZZ2196=ZZ2196+ZZ2196
+006120	014250		0 8192 -ZZ1196
+006121	673400		0 ZZ2196
-	 mark 1887, 480		/123 taur
+006122	001700		ZZ2197=ZZ2197+ZZ2197
+006122	003600		ZZ2197=ZZ2197+ZZ2197
+006122	007400		ZZ2197=ZZ2197+ZZ2197
+006122	017000		ZZ2197=ZZ2197+ZZ2197
+006122	036000		ZZ2197=ZZ2197+ZZ2197
+006122	074000		ZZ2197=ZZ2197+ZZ2197
+006122	170000		ZZ2197=ZZ2197+ZZ2197
+006122	360000		ZZ2197=ZZ2197+ZZ2197
+006122	014241		0 8192 -ZZ1197
+006123	360000		0 ZZ2197
-	 mark 1948, -338	/14 leps
+006124	776532		ZZ2198=ZZ2198+ZZ2198
+006124	775264		ZZ2198=ZZ2198+ZZ2198
+006124	772550		ZZ2198=ZZ2198+ZZ2198
+006124	765320		ZZ2198=ZZ2198+ZZ2198
+006124	752640		ZZ2198=ZZ2198+ZZ2198
+006124	725500		ZZ2198=ZZ2198+ZZ2198
+006124	653200		ZZ2198=ZZ2198+ZZ2198
+006124	526400		ZZ2198=ZZ2198+ZZ2198
+006124	014144		0 8192 -ZZ1198
+006125	526400		0 ZZ2198
-	 mark 2274, 296		/31 gemi
+006126	001120		ZZ2199=ZZ2199+ZZ2199
+006126	002240		ZZ2199=ZZ2199+ZZ2199
+006126	004500		ZZ2199=ZZ2199+ZZ2199
+006126	011200		ZZ2199=ZZ2199+ZZ2199
+006126	022400		ZZ2199=ZZ2199+ZZ2199
+006126	045000		ZZ2199=ZZ2199+ZZ2199
+006126	112000		ZZ2199=ZZ2199+ZZ2199
+006126	224000		ZZ2199=ZZ2199+ZZ2199
+006126	013436		0 8192 -ZZ1199
+006127	224000		0 ZZ2199
-	 mark 2460, 380		/54 gemi
+006130	001370		ZZ2200=ZZ2200+ZZ2200
+006130	002760		ZZ2200=ZZ2200+ZZ2200
+006130	005740		ZZ2200=ZZ2200+ZZ2200
+006130	013700		ZZ2200=ZZ2200+ZZ2200
+006130	027600		ZZ2200=ZZ2200+ZZ2200
+006130	057400		ZZ2200=ZZ2200+ZZ2200
+006130	137000		ZZ2200=ZZ2200+ZZ2200
+006130	276000		ZZ2200=ZZ2200+ZZ2200
+006130	013144		0 8192 -ZZ1200
+006131	276000		0 ZZ2200
-	 mark 2470, 504		/55 gemi
+006132	001760		ZZ2201=ZZ2201+ZZ2201
+006132	003740		ZZ2201=ZZ2201+ZZ2201
+006132	007700		ZZ2201=ZZ2201+ZZ2201
+006132	017600		ZZ2201=ZZ2201+ZZ2201
+006132	037400		ZZ2201=ZZ2201+ZZ2201
+006132	077000		ZZ2201=ZZ2201+ZZ2201
+006132	176000		ZZ2201=ZZ2201+ZZ2201
+006132	374000		ZZ2201=ZZ2201+ZZ2201
+006132	013132		0 8192 -ZZ1201
+006133	374000		0 ZZ2201
-	 mark 2513, 193		/ 3 cmin
+006134	000602		ZZ2202=ZZ2202+ZZ2202
+006134	001404		ZZ2202=ZZ2202+ZZ2202
+006134	003010		ZZ2202=ZZ2202+ZZ2202
+006134	006020		ZZ2202=ZZ2202+ZZ2202
+006134	014040		ZZ2202=ZZ2202+ZZ2202
+006134	030100		ZZ2202=ZZ2202+ZZ2202
+006134	060200		ZZ2202=ZZ2202+ZZ2202
+006134	140400		ZZ2202=ZZ2202+ZZ2202
+006134	013057		0 8192 -ZZ1202
+006135	140400		0 ZZ2202
-	 mark 2967, 154		/11 hyda
+006136	000464		ZZ2203=ZZ2203+ZZ2203
+006136	001150		ZZ2203=ZZ2203+ZZ2203
+006136	002320		ZZ2203=ZZ2203+ZZ2203
+006136	004640		ZZ2203=ZZ2203+ZZ2203
+006136	011500		ZZ2203=ZZ2203+ZZ2203
+006136	023200		ZZ2203=ZZ2203+ZZ2203
+006136	046400		ZZ2203=ZZ2203+ZZ2203
+006136	115000		ZZ2203=ZZ2203+ZZ2203
+006136	012151		0 8192 -ZZ1203
+006137	115000		0 ZZ2203
-	 mark 3016, 144		/16 hyda
+006140	000440		ZZ2204=ZZ2204+ZZ2204
+006140	001100		ZZ2204=ZZ2204+ZZ2204
+006140	002200		ZZ2204=ZZ2204+ZZ2204
+006140	004400		ZZ2204=ZZ2204+ZZ2204
+006140	011000		ZZ2204=ZZ2204+ZZ2204
+006140	022000		ZZ2204=ZZ2204+ZZ2204
+006140	044000		ZZ2204=ZZ2204+ZZ2204
+006140	110000		ZZ2204=ZZ2204+ZZ2204
+006140	012070		0 8192 -ZZ1204
+006141	110000		0 ZZ2204
-	 mark 3424, 393		/30 leon
+006142	001422		ZZ2205=ZZ2205+ZZ2205
+006142	003044		ZZ2205=ZZ2205+ZZ2205
+006142	006110		ZZ2205=ZZ2205+ZZ2205
+006142	014220		ZZ2205=ZZ2205+ZZ2205
+006142	030440		ZZ2205=ZZ2205+ZZ2205
+006142	061100		ZZ2205=ZZ2205+ZZ2205
+006142	142200		ZZ2205=ZZ2205+ZZ2205
+006142	304400		ZZ2205=ZZ2205+ZZ2205
+006142	011240		0 8192 -ZZ1205
+006143	304400		0 ZZ2205
-	 mark 3496, 463		/41 leon, algieba
+006144	001636		ZZ2206=ZZ2206+ZZ2206
+006144	003474		ZZ2206=ZZ2206+ZZ2206
+006144	007170		ZZ2206=ZZ2206+ZZ2206
+006144	016360		ZZ2206=ZZ2206+ZZ2206
+006144	034740		ZZ2206=ZZ2206+ZZ2206
+006144	071700		ZZ2206=ZZ2206+ZZ2206
+006144	163600		ZZ2206=ZZ2206+ZZ2206
+006144	347400		ZZ2206=ZZ2206+ZZ2206
+006144	011130		0 8192 -ZZ1206
+006145	347400		0 ZZ2206
-	 mark 3668, -357	/nu hyda
+006146	776464		ZZ2207=ZZ2207+ZZ2207
+006146	775150		ZZ2207=ZZ2207+ZZ2207
+006146	772320		ZZ2207=ZZ2207+ZZ2207
+006146	764640		ZZ2207=ZZ2207+ZZ2207
+006146	751500		ZZ2207=ZZ2207+ZZ2207
+006146	723200		ZZ2207=ZZ2207+ZZ2207
+006146	646400		ZZ2207=ZZ2207+ZZ2207
+006146	515000		ZZ2207=ZZ2207+ZZ2207
+006146	010654		0 8192 -ZZ1207
+006147	515000		0 ZZ2207
-	 mark 3805, 479		/68 leon
+006150	001676		ZZ2208=ZZ2208+ZZ2208
+006150	003574		ZZ2208=ZZ2208+ZZ2208
+006150	007370		ZZ2208=ZZ2208+ZZ2208
+006150	016760		ZZ2208=ZZ2208+ZZ2208
+006150	035740		ZZ2208=ZZ2208+ZZ2208
+006150	073700		ZZ2208=ZZ2208+ZZ2208
+006150	167600		ZZ2208=ZZ2208+ZZ2208
+006150	357400		ZZ2208=ZZ2208+ZZ2208
+006150	010443		0 8192 -ZZ1208
+006151	357400		0 ZZ2208
-	 mark 3806, 364		/10 leon
+006152	001330		ZZ2209=ZZ2209+ZZ2209
+006152	002660		ZZ2209=ZZ2209+ZZ2209
+006152	005540		ZZ2209=ZZ2209+ZZ2209
+006152	013300		ZZ2209=ZZ2209+ZZ2209
+006152	026600		ZZ2209=ZZ2209+ZZ2209
+006152	055400		ZZ2209=ZZ2209+ZZ2209
+006152	133000		ZZ2209=ZZ2209+ZZ2209
+006152	266000		ZZ2209=ZZ2209+ZZ2209
+006152	010442		0 8192 -ZZ1209
+006153	266000		0 ZZ2209
-	 mark 4124, -502	/ 2 corv
+006154	776022		ZZ2210=ZZ2210+ZZ2210
+006154	774044		ZZ2210=ZZ2210+ZZ2210
+006154	770110		ZZ2210=ZZ2210+ZZ2210
+006154	760220		ZZ2210=ZZ2210+ZZ2210
+006154	740440		ZZ2210=ZZ2210+ZZ2210
+006154	701100		ZZ2210=ZZ2210+ZZ2210
+006154	602200		ZZ2210=ZZ2210+ZZ2210
+006154	404400		ZZ2210=ZZ2210+ZZ2210
+006154	007744		0 8192 -ZZ1210
+006155	404400		0 ZZ2210
-	 mark 4157, -387	/ 4 corv
+006156	776370		ZZ2211=ZZ2211+ZZ2211
+006156	774760		ZZ2211=ZZ2211+ZZ2211
+006156	771740		ZZ2211=ZZ2211+ZZ2211
+006156	763700		ZZ2211=ZZ2211+ZZ2211
+006156	747600		ZZ2211=ZZ2211+ZZ2211
+006156	717400		ZZ2211=ZZ2211+ZZ2211
+006156	637000		ZZ2211=ZZ2211+ZZ2211
+006156	476000		ZZ2211=ZZ2211+ZZ2211
+006156	007703		0 8192 -ZZ1211
+006157	476000		0 ZZ2211
-	 mark 4236, -363	/ 7 corv
+006160	776450		ZZ2212=ZZ2212+ZZ2212
+006160	775120		ZZ2212=ZZ2212+ZZ2212
+006160	772240		ZZ2212=ZZ2212+ZZ2212
+006160	764500		ZZ2212=ZZ2212+ZZ2212
+006160	751200		ZZ2212=ZZ2212+ZZ2212
+006160	722400		ZZ2212=ZZ2212+ZZ2212
+006160	645000		ZZ2212=ZZ2212+ZZ2212
+006160	512000		ZZ2212=ZZ2212+ZZ2212
+006160	007564		0 8192 -ZZ1212
+006161	512000		0 ZZ2212
-	 mark 4304, -21		/29 virg
+006162	777724		ZZ2213=ZZ2213+ZZ2213
+006162	777650		ZZ2213=ZZ2213+ZZ2213
+006162	777520		ZZ2213=ZZ2213+ZZ2213
+006162	777240		ZZ2213=ZZ2213+ZZ2213
+006162	776500		ZZ2213=ZZ2213+ZZ2213
+006162	775200		ZZ2213=ZZ2213+ZZ2213
+006162	772400		ZZ2213=ZZ2213+ZZ2213
+006162	765000		ZZ2213=ZZ2213+ZZ2213
+006162	007460		0 8192 -ZZ1213
+006163	765000		0 ZZ2213
-	 mark 4384, 90		/43 virg
+006164	000264		ZZ2214=ZZ2214+ZZ2214
+006164	000550		ZZ2214=ZZ2214+ZZ2214
+006164	001320		ZZ2214=ZZ2214+ZZ2214
+006164	002640		ZZ2214=ZZ2214+ZZ2214
+006164	005500		ZZ2214=ZZ2214+ZZ2214
+006164	013200		ZZ2214=ZZ2214+ZZ2214
+006164	026400		ZZ2214=ZZ2214+ZZ2214
+006164	055000		ZZ2214=ZZ2214+ZZ2214
+006164	007340		0 8192 -ZZ1214
+006165	055000		0 ZZ2214
-	 mark 4421, 262		/47 virg
+006166	001014		ZZ2215=ZZ2215+ZZ2215
+006166	002030		ZZ2215=ZZ2215+ZZ2215
+006166	004060		ZZ2215=ZZ2215+ZZ2215
+006166	010140		ZZ2215=ZZ2215+ZZ2215
+006166	020300		ZZ2215=ZZ2215+ZZ2215
+006166	040600		ZZ2215=ZZ2215+ZZ2215
+006166	101400		ZZ2215=ZZ2215+ZZ2215
+006166	203000		ZZ2215=ZZ2215+ZZ2215
+006166	007273		0 8192 -ZZ1215
+006167	203000		0 ZZ2215
-	 mark 4606, -2		/79 virg
+006170	777772		ZZ2216=ZZ2216+ZZ2216
+006170	777764		ZZ2216=ZZ2216+ZZ2216
+006170	777750		ZZ2216=ZZ2216+ZZ2216
+006170	777720		ZZ2216=ZZ2216+ZZ2216
+006170	777640		ZZ2216=ZZ2216+ZZ2216
+006170	777500		ZZ2216=ZZ2216+ZZ2216
+006170	777200		ZZ2216=ZZ2216+ZZ2216
+006170	776400		ZZ2216=ZZ2216+ZZ2216
+006170	007002		0 8192 -ZZ1216
+006171	776400		0 ZZ2216
-	 mark 4721, 430		/ 8 boot
+006172	001534		ZZ2217=ZZ2217+ZZ2217
+006172	003270		ZZ2217=ZZ2217+ZZ2217
+006172	006560		ZZ2217=ZZ2217+ZZ2217
+006172	015340		ZZ2217=ZZ2217+ZZ2217
+006172	032700		ZZ2217=ZZ2217+ZZ2217
+006172	065600		ZZ2217=ZZ2217+ZZ2217
+006172	153400		ZZ2217=ZZ2217+ZZ2217
+006172	327000		ZZ2217=ZZ2217+ZZ2217
+006172	006617		0 8192 -ZZ1217
+006173	327000		0 ZZ2217
-	 mark 5037, -356	/ 9 libr
+006174	776466		ZZ2218=ZZ2218+ZZ2218
+006174	775154		ZZ2218=ZZ2218+ZZ2218
+006174	772330		ZZ2218=ZZ2218+ZZ2218
+006174	764660		ZZ2218=ZZ2218+ZZ2218
+006174	751540		ZZ2218=ZZ2218+ZZ2218
+006174	723300		ZZ2218=ZZ2218+ZZ2218
+006174	646600		ZZ2218=ZZ2218+ZZ2218
+006174	515400		ZZ2218=ZZ2218+ZZ2218
+006174	006123		0 8192 -ZZ1218
+006175	515400		0 ZZ2218
-	 mark 5186, -205	/27 libr
+006176	777144		ZZ2219=ZZ2219+ZZ2219
+006176	776310		ZZ2219=ZZ2219+ZZ2219
+006176	774620		ZZ2219=ZZ2219+ZZ2219
+006176	771440		ZZ2219=ZZ2219+ZZ2219
+006176	763100		ZZ2219=ZZ2219+ZZ2219
+006176	746200		ZZ2219=ZZ2219+ZZ2219
+006176	714400		ZZ2219=ZZ2219+ZZ2219
+006176	631000		ZZ2219=ZZ2219+ZZ2219
+006176	005676		0 8192 -ZZ1219
+006177	631000		0 ZZ2219
-	 mark 5344, 153		/24 serp
+006200	000462		ZZ2220=ZZ2220+ZZ2220
+006200	001144		ZZ2220=ZZ2220+ZZ2220
+006200	002310		ZZ2220=ZZ2220+ZZ2220
+006200	004620		ZZ2220=ZZ2220+ZZ2220
+006200	011440		ZZ2220=ZZ2220+ZZ2220
+006200	023100		ZZ2220=ZZ2220+ZZ2220
+006200	046200		ZZ2220=ZZ2220+ZZ2220
+006200	114400		ZZ2220=ZZ2220+ZZ2220
+006200	005440		0 8192 -ZZ1220
+006201	114400		0 ZZ2220
-	 mark 5357, 358		/28 serp
+006202	001314		ZZ2221=ZZ2221+ZZ2221
+006202	002630		ZZ2221=ZZ2221+ZZ2221
+006202	005460		ZZ2221=ZZ2221+ZZ2221
+006202	013140		ZZ2221=ZZ2221+ZZ2221
+006202	026300		ZZ2221=ZZ2221+ZZ2221
+006202	054600		ZZ2221=ZZ2221+ZZ2221
+006202	131400		ZZ2221=ZZ2221+ZZ2221
+006202	263000		ZZ2221=ZZ2221+ZZ2221
+006202	005423		0 8192 -ZZ1221
+006203	263000		0 ZZ2221
-	 mark 5373, -71		/32 serp
+006204	777560		ZZ2222=ZZ2222+ZZ2222
+006204	777340		ZZ2222=ZZ2222+ZZ2222
+006204	776700		ZZ2222=ZZ2222+ZZ2222
+006204	775600		ZZ2222=ZZ2222+ZZ2222
+006204	773400		ZZ2222=ZZ2222+ZZ2222
+006204	767000		ZZ2222=ZZ2222+ZZ2222
+006204	756000		ZZ2222=ZZ2222+ZZ2222
+006204	734000		ZZ2222=ZZ2222+ZZ2222
+006204	005403		0 8192 -ZZ1222
+006205	734000		0 ZZ2222
-	 mark 5430, -508	/ 7 scor
+006206	776006		ZZ2223=ZZ2223+ZZ2223
+006206	774014		ZZ2223=ZZ2223+ZZ2223
+006206	770030		ZZ2223=ZZ2223+ZZ2223
+006206	760060		ZZ2223=ZZ2223+ZZ2223
+006206	740140		ZZ2223=ZZ2223+ZZ2223
+006206	700300		ZZ2223=ZZ2223+ZZ2223
+006206	600600		ZZ2223=ZZ2223+ZZ2223
+006206	401400		ZZ2223=ZZ2223+ZZ2223
+006206	005312		0 8192 -ZZ1223
+006207	401400		0 ZZ2223
-	 mark 5459, -445	/ 8 scor
+006210	776204		ZZ2224=ZZ2224+ZZ2224
+006210	774410		ZZ2224=ZZ2224+ZZ2224
+006210	771020		ZZ2224=ZZ2224+ZZ2224
+006210	762040		ZZ2224=ZZ2224+ZZ2224
+006210	744100		ZZ2224=ZZ2224+ZZ2224
+006210	710200		ZZ2224=ZZ2224+ZZ2224
+006210	620400		ZZ2224=ZZ2224+ZZ2224
+006210	441000		ZZ2224=ZZ2224+ZZ2224
+006210	005255		0 8192 -ZZ1224
+006211	441000		0 ZZ2224
-	 mark 5513, -78		/ 1 ophi
+006212	777542		ZZ2225=ZZ2225+ZZ2225
+006212	777304		ZZ2225=ZZ2225+ZZ2225
+006212	776610		ZZ2225=ZZ2225+ZZ2225
+006212	775420		ZZ2225=ZZ2225+ZZ2225
+006212	773040		ZZ2225=ZZ2225+ZZ2225
+006212	766100		ZZ2225=ZZ2225+ZZ2225
+006212	754200		ZZ2225=ZZ2225+ZZ2225
+006212	730400		ZZ2225=ZZ2225+ZZ2225
+006212	005167		0 8192 -ZZ1225
+006213	730400		0 ZZ2225
-	 mark 5536, -101	/ 2 ophi
+006214	777464		ZZ2226=ZZ2226+ZZ2226
+006214	777150		ZZ2226=ZZ2226+ZZ2226
+006214	776320		ZZ2226=ZZ2226+ZZ2226
+006214	774640		ZZ2226=ZZ2226+ZZ2226
+006214	771500		ZZ2226=ZZ2226+ZZ2226
+006214	763200		ZZ2226=ZZ2226+ZZ2226
+006214	746400		ZZ2226=ZZ2226+ZZ2226
+006214	715000		ZZ2226=ZZ2226+ZZ2226
+006214	005140		0 8192 -ZZ1226
+006215	715000		0 ZZ2226
-	 mark 5609, 494		/27 herc
+006216	001734		ZZ2227=ZZ2227+ZZ2227
+006216	003670		ZZ2227=ZZ2227+ZZ2227
+006216	007560		ZZ2227=ZZ2227+ZZ2227
+006216	017340		ZZ2227=ZZ2227+ZZ2227
+006216	036700		ZZ2227=ZZ2227+ZZ2227
+006216	075600		ZZ2227=ZZ2227+ZZ2227
+006216	173400		ZZ2227=ZZ2227+ZZ2227
+006216	367000		ZZ2227=ZZ2227+ZZ2227
+006216	005027		0 8192 -ZZ1227
+006217	367000		0 ZZ2227
-	 mark 5641, -236	/13 ophi
+006220	777046		ZZ2228=ZZ2228+ZZ2228
+006220	776114		ZZ2228=ZZ2228+ZZ2228
+006220	774230		ZZ2228=ZZ2228+ZZ2228
+006220	770460		ZZ2228=ZZ2228+ZZ2228
+006220	761140		ZZ2228=ZZ2228+ZZ2228
+006220	742300		ZZ2228=ZZ2228+ZZ2228
+006220	704600		ZZ2228=ZZ2228+ZZ2228
+006220	611400		ZZ2228=ZZ2228+ZZ2228
+006220	004767		0 8192 -ZZ1228
+006221	611400		0 ZZ2228
-	 mark 5828, -355	/35 ophi
+006222	776470		ZZ2229=ZZ2229+ZZ2229
+006222	775160		ZZ2229=ZZ2229+ZZ2229
+006222	772340		ZZ2229=ZZ2229+ZZ2229
+006222	764700		ZZ2229=ZZ2229+ZZ2229
+006222	751600		ZZ2229=ZZ2229+ZZ2229
+006222	723400		ZZ2229=ZZ2229+ZZ2229
+006222	647000		ZZ2229=ZZ2229+ZZ2229
+006222	516000		ZZ2229=ZZ2229+ZZ2229
+006222	004474		0 8192 -ZZ1229
+006223	516000		0 ZZ2229
-	 mark 5860, 330		/64 herc
+006224	001224		ZZ2230=ZZ2230+ZZ2230
+006224	002450		ZZ2230=ZZ2230+ZZ2230
+006224	005120		ZZ2230=ZZ2230+ZZ2230
+006224	012240		ZZ2230=ZZ2230+ZZ2230
+006224	024500		ZZ2230=ZZ2230+ZZ2230
+006224	051200		ZZ2230=ZZ2230+ZZ2230
+006224	122400		ZZ2230=ZZ2230+ZZ2230
+006224	245000		ZZ2230=ZZ2230+ZZ2230
+006224	004434		0 8192 -ZZ1230
+006225	245000		0 ZZ2230
-	 mark 5984, -349	/55 serp
+006226	776504		ZZ2231=ZZ2231+ZZ2231
+006226	775210		ZZ2231=ZZ2231+ZZ2231
+006226	772420		ZZ2231=ZZ2231+ZZ2231
+006226	765040		ZZ2231=ZZ2231+ZZ2231
+006226	752100		ZZ2231=ZZ2231+ZZ2231
+006226	724200		ZZ2231=ZZ2231+ZZ2231
+006226	650400		ZZ2231=ZZ2231+ZZ2231
+006226	521000		ZZ2231=ZZ2231+ZZ2231
+006226	004240		0 8192 -ZZ1231
+006227	521000		0 ZZ2231
-	 mark 6047, 63		/62 ophi
+006230	000176		ZZ2232=ZZ2232+ZZ2232
+006230	000374		ZZ2232=ZZ2232+ZZ2232
+006230	000770		ZZ2232=ZZ2232+ZZ2232
+006230	001760		ZZ2232=ZZ2232+ZZ2232
+006230	003740		ZZ2232=ZZ2232+ZZ2232
+006230	007700		ZZ2232=ZZ2232+ZZ2232
+006230	017600		ZZ2232=ZZ2232+ZZ2232
+006230	037400		ZZ2232=ZZ2232+ZZ2232
+006230	004141		0 8192 -ZZ1232
+006231	037400		0 ZZ2232
-	 mark 6107, -222	/64 ophi
+006232	777102		ZZ2233=ZZ2233+ZZ2233
+006232	776204		ZZ2233=ZZ2233+ZZ2233
+006232	774410		ZZ2233=ZZ2233+ZZ2233
+006232	771020		ZZ2233=ZZ2233+ZZ2233
+006232	762040		ZZ2233=ZZ2233+ZZ2233
+006232	744100		ZZ2233=ZZ2233+ZZ2233
+006232	710200		ZZ2233=ZZ2233+ZZ2233
+006232	620400		ZZ2233=ZZ2233+ZZ2233
+006232	004045		0 8192 -ZZ1233
+006233	620400		0 ZZ2233
-	 mark 6159, 217		/72 ophi
+006234	000662		ZZ2234=ZZ2234+ZZ2234
+006234	001544		ZZ2234=ZZ2234+ZZ2234
+006234	003310		ZZ2234=ZZ2234+ZZ2234
+006234	006620		ZZ2234=ZZ2234+ZZ2234
+006234	015440		ZZ2234=ZZ2234+ZZ2234
+006234	033100		ZZ2234=ZZ2234+ZZ2234
+006234	066200		ZZ2234=ZZ2234+ZZ2234
+006234	154400		ZZ2234=ZZ2234+ZZ2234
+006234	003761		0 8192 -ZZ1234
+006235	154400		0 ZZ2234
-	 mark 6236, -66		/58 serp
+006236	777572		ZZ2235=ZZ2235+ZZ2235
+006236	777364		ZZ2235=ZZ2235+ZZ2235
+006236	776750		ZZ2235=ZZ2235+ZZ2235
+006236	775720		ZZ2235=ZZ2235+ZZ2235
+006236	773640		ZZ2235=ZZ2235+ZZ2235
+006236	767500		ZZ2235=ZZ2235+ZZ2235
+006236	757200		ZZ2235=ZZ2235+ZZ2235
+006236	736400		ZZ2235=ZZ2235+ZZ2235
+006236	003644		0 8192 -ZZ1235
+006237	736400		0 ZZ2235
-	 mark 6439, -483        /37 sgtr
+006240	776070		ZZ2236=ZZ2236+ZZ2236
+006240	774160		ZZ2236=ZZ2236+ZZ2236
+006240	770340		ZZ2236=ZZ2236+ZZ2236
+006240	760700		ZZ2236=ZZ2236+ZZ2236
+006240	741600		ZZ2236=ZZ2236+ZZ2236
+006240	703400		ZZ2236=ZZ2236+ZZ2236
+006240	607000		ZZ2236=ZZ2236+ZZ2236
+006240	416000		ZZ2236=ZZ2236+ZZ2236
+006240	003331		0 8192 -ZZ1236
+006241	416000		0 ZZ2236
-	 mark 6490, 312         /17 aqil
+006242	001160		ZZ2237=ZZ2237+ZZ2237
+006242	002340		ZZ2237=ZZ2237+ZZ2237
+006242	004700		ZZ2237=ZZ2237+ZZ2237
+006242	011600		ZZ2237=ZZ2237+ZZ2237
+006242	023400		ZZ2237=ZZ2237+ZZ2237
+006242	047000		ZZ2237=ZZ2237+ZZ2237
+006242	116000		ZZ2237=ZZ2237+ZZ2237
+006242	234000		ZZ2237=ZZ2237+ZZ2237
+006242	003246		0 8192 -ZZ1237
+006243	234000		0 ZZ2237
-	 mark 6491, -115        /16 aqil
+006244	777430		ZZ2238=ZZ2238+ZZ2238
+006244	777060		ZZ2238=ZZ2238+ZZ2238
+006244	776140		ZZ2238=ZZ2238+ZZ2238
+006244	774300		ZZ2238=ZZ2238+ZZ2238
+006244	770600		ZZ2238=ZZ2238+ZZ2238
+006244	761400		ZZ2238=ZZ2238+ZZ2238
+006244	743000		ZZ2238=ZZ2238+ZZ2238
+006244	706000		ZZ2238=ZZ2238+ZZ2238
+006244	003245		0 8192 -ZZ1238
+006245	706000		0 ZZ2238
-	 mark 6507, -482        /41 sgtr
+006246	776072		ZZ2239=ZZ2239+ZZ2239
+006246	774164		ZZ2239=ZZ2239+ZZ2239
+006246	770350		ZZ2239=ZZ2239+ZZ2239
+006246	760720		ZZ2239=ZZ2239+ZZ2239
+006246	741640		ZZ2239=ZZ2239+ZZ2239
+006246	703500		ZZ2239=ZZ2239+ZZ2239
+006246	607200		ZZ2239=ZZ2239+ZZ2239
+006246	416400		ZZ2239=ZZ2239+ZZ2239
+006246	003225		0 8192 -ZZ1239
+006247	416400		0 ZZ2239
-	 mark 6602, 66          /30 aqil
+006250	000204		ZZ2240=ZZ2240+ZZ2240
+006250	000410		ZZ2240=ZZ2240+ZZ2240
+006250	001020		ZZ2240=ZZ2240+ZZ2240
+006250	002040		ZZ2240=ZZ2240+ZZ2240
+006250	004100		ZZ2240=ZZ2240+ZZ2240
+006250	010200		ZZ2240=ZZ2240+ZZ2240
+006250	020400		ZZ2240=ZZ2240+ZZ2240
+006250	041000		ZZ2240=ZZ2240+ZZ2240
+006250	003066		0 8192 -ZZ1240
+006251	041000		0 ZZ2240
-	 mark 6721, 236         /50 aqil
+006252	000730		ZZ2241=ZZ2241+ZZ2241
+006252	001660		ZZ2241=ZZ2241+ZZ2241
+006252	003540		ZZ2241=ZZ2241+ZZ2241
+006252	007300		ZZ2241=ZZ2241+ZZ2241
+006252	016600		ZZ2241=ZZ2241+ZZ2241
+006252	035400		ZZ2241=ZZ2241+ZZ2241
+006252	073000		ZZ2241=ZZ2241+ZZ2241
+006252	166000		ZZ2241=ZZ2241+ZZ2241
+006252	002677		0 8192 -ZZ1241
+006253	166000		0 ZZ2241
-	 mark 6794, 437         /12 sgte
+006254	001552		ZZ2242=ZZ2242+ZZ2242
+006254	003324		ZZ2242=ZZ2242+ZZ2242
+006254	006650		ZZ2242=ZZ2242+ZZ2242
+006254	015520		ZZ2242=ZZ2242+ZZ2242
+006254	033240		ZZ2242=ZZ2242+ZZ2242
+006254	066500		ZZ2242=ZZ2242+ZZ2242
+006254	155200		ZZ2242=ZZ2242+ZZ2242
+006254	332400		ZZ2242=ZZ2242+ZZ2242
+006254	002566		0 8192 -ZZ1242
+006255	332400		0 ZZ2242
-	 mark 6862, -25         /65 aqil
+006256	777714		ZZ2243=ZZ2243+ZZ2243
+006256	777630		ZZ2243=ZZ2243+ZZ2243
+006256	777460		ZZ2243=ZZ2243+ZZ2243
+006256	777140		ZZ2243=ZZ2243+ZZ2243
+006256	776300		ZZ2243=ZZ2243+ZZ2243
+006256	774600		ZZ2243=ZZ2243+ZZ2243
+006256	771400		ZZ2243=ZZ2243+ZZ2243
+006256	763000		ZZ2243=ZZ2243+ZZ2243
+006256	002462		0 8192 -ZZ1243
+006257	763000		0 ZZ2243
-	 mark 6914, -344        / 9 capr
+006260	776516		ZZ2244=ZZ2244+ZZ2244
+006260	775234		ZZ2244=ZZ2244+ZZ2244
+006260	772470		ZZ2244=ZZ2244+ZZ2244
+006260	765160		ZZ2244=ZZ2244+ZZ2244
+006260	752340		ZZ2244=ZZ2244+ZZ2244
+006260	724700		ZZ2244=ZZ2244+ZZ2244
+006260	651600		ZZ2244=ZZ2244+ZZ2244
+006260	523400		ZZ2244=ZZ2244+ZZ2244
+006260	002376		0 8192 -ZZ1244
+006261	523400		0 ZZ2244
-	 mark 7014, 324         / 6 dlph
+006262	001210		ZZ2245=ZZ2245+ZZ2245
+006262	002420		ZZ2245=ZZ2245+ZZ2245
+006262	005040		ZZ2245=ZZ2245+ZZ2245
+006262	012100		ZZ2245=ZZ2245+ZZ2245
+006262	024200		ZZ2245=ZZ2245+ZZ2245
+006262	050400		ZZ2245=ZZ2245+ZZ2245
+006262	121000		ZZ2245=ZZ2245+ZZ2245
+006262	242000		ZZ2245=ZZ2245+ZZ2245
+006262	002232		0 8192 -ZZ1245
+006263	242000		0 ZZ2245
-	 mark 7318, -137        /22 aqar
+006264	777354		ZZ2246=ZZ2246+ZZ2246
+006264	776730		ZZ2246=ZZ2246+ZZ2246
+006264	775660		ZZ2246=ZZ2246+ZZ2246
+006264	773540		ZZ2246=ZZ2246+ZZ2246
+006264	767300		ZZ2246=ZZ2246+ZZ2246
+006264	756600		ZZ2246=ZZ2246+ZZ2246
+006264	735400		ZZ2246=ZZ2246+ZZ2246
+006264	673000		ZZ2246=ZZ2246+ZZ2246
+006264	001552		0 8192 -ZZ1246
+006265	673000		0 ZZ2246
-	 mark 7391, 214         / 8 pegs
+006266	000654		ZZ2247=ZZ2247+ZZ2247
+006266	001530		ZZ2247=ZZ2247+ZZ2247
+006266	003260		ZZ2247=ZZ2247+ZZ2247
+006266	006540		ZZ2247=ZZ2247+ZZ2247
+006266	015300		ZZ2247=ZZ2247+ZZ2247
+006266	032600		ZZ2247=ZZ2247+ZZ2247
+006266	065400		ZZ2247=ZZ2247+ZZ2247
+006266	153000		ZZ2247=ZZ2247+ZZ2247
+006266	001441		0 8192 -ZZ1247
+006267	153000		0 ZZ2247
-	 mark 7404, -377        /49 capr
+006270	776414		ZZ2248=ZZ2248+ZZ2248
+006270	775030		ZZ2248=ZZ2248+ZZ2248
+006270	772060		ZZ2248=ZZ2248+ZZ2248
+006270	764140		ZZ2248=ZZ2248+ZZ2248
+006270	750300		ZZ2248=ZZ2248+ZZ2248
+006270	720600		ZZ2248=ZZ2248+ZZ2248
+006270	641400		ZZ2248=ZZ2248+ZZ2248
+006270	503000		ZZ2248=ZZ2248+ZZ2248
+006270	001424		0 8192 -ZZ1248
+006271	503000		0 ZZ2248
-	 mark 7513, -18         /34 aqar
+006272	777732		ZZ2249=ZZ2249+ZZ2249
+006272	777664		ZZ2249=ZZ2249+ZZ2249
+006272	777550		ZZ2249=ZZ2249+ZZ2249
+006272	777320		ZZ2249=ZZ2249+ZZ2249
+006272	776640		ZZ2249=ZZ2249+ZZ2249
+006272	775500		ZZ2249=ZZ2249+ZZ2249
+006272	773200		ZZ2249=ZZ2249+ZZ2249
+006272	766400		ZZ2249=ZZ2249+ZZ2249
+006272	001247		0 8192 -ZZ1249
+006273	766400		0 ZZ2249
-	 mark 7539, 130         /26 pegs
+006274	000404		ZZ2250=ZZ2250+ZZ2250
+006274	001010		ZZ2250=ZZ2250+ZZ2250
+006274	002020		ZZ2250=ZZ2250+ZZ2250
+006274	004040		ZZ2250=ZZ2250+ZZ2250
+006274	010100		ZZ2250=ZZ2250+ZZ2250
+006274	020200		ZZ2250=ZZ2250+ZZ2250
+006274	040400		ZZ2250=ZZ2250+ZZ2250
+006274	101000		ZZ2250=ZZ2250+ZZ2250
+006274	001215		0 8192 -ZZ1250
+006275	101000		0 ZZ2250
-	 mark 7644, -12         /55 aqar
+006276	777746		ZZ2251=ZZ2251+ZZ2251
+006276	777714		ZZ2251=ZZ2251+ZZ2251
+006276	777630		ZZ2251=ZZ2251+ZZ2251
+006276	777460		ZZ2251=ZZ2251+ZZ2251
+006276	777140		ZZ2251=ZZ2251+ZZ2251
+006276	776300		ZZ2251=ZZ2251+ZZ2251
+006276	774600		ZZ2251=ZZ2251+ZZ2251
+006276	771400		ZZ2251=ZZ2251+ZZ2251
+006276	001044		0 8192 -ZZ1251
+006277	771400		0 ZZ2251
-	 mark 7717, 235         /42 pegs
+006300	000726		ZZ2252=ZZ2252+ZZ2252
+006300	001654		ZZ2252=ZZ2252+ZZ2252
+006300	003530		ZZ2252=ZZ2252+ZZ2252
+006300	007260		ZZ2252=ZZ2252+ZZ2252
+006300	016540		ZZ2252=ZZ2252+ZZ2252
+006300	035300		ZZ2252=ZZ2252+ZZ2252
+006300	072600		ZZ2252=ZZ2252+ZZ2252
+006300	165400		ZZ2252=ZZ2252+ZZ2252
+006300	000733		0 8192 -ZZ1252
+006301	165400		0 ZZ2252
-	 mark 7790, -372        /76 aqar
+006302	776426		ZZ2253=ZZ2253+ZZ2253
+006302	775054		ZZ2253=ZZ2253+ZZ2253
+006302	772130		ZZ2253=ZZ2253+ZZ2253
+006302	764260		ZZ2253=ZZ2253+ZZ2253
+006302	750540		ZZ2253=ZZ2253+ZZ2253
+006302	721300		ZZ2253=ZZ2253+ZZ2253
+006302	642600		ZZ2253=ZZ2253+ZZ2253
+006302	505400		ZZ2253=ZZ2253+ZZ2253
+006302	000622		0 8192 -ZZ1253
+006303	505400		0 ZZ2253
 006304		3q,
-	 mark 7849, 334		/54 pegs, markab
+006304	001234		ZZ2254=ZZ2254+ZZ2254
+006304	002470		ZZ2254=ZZ2254+ZZ2254
+006304	005160		ZZ2254=ZZ2254+ZZ2254
+006304	012340		ZZ2254=ZZ2254+ZZ2254
+006304	024700		ZZ2254=ZZ2254+ZZ2254
+006304	051600		ZZ2254=ZZ2254+ZZ2254
+006304	123400		ZZ2254=ZZ2254+ZZ2254
+006304	247000		ZZ2254=ZZ2254+ZZ2254
+006304	000527		0 8192 -ZZ1254
+006305	247000		0 ZZ2254
 006306		4j,
- 	 mark 1, -143 		/33 pisc
+006306	777340		ZZ2255=ZZ2255+ZZ2255
+006306	776700		ZZ2255=ZZ2255+ZZ2255
+006306	775600		ZZ2255=ZZ2255+ZZ2255
+006306	773400		ZZ2255=ZZ2255+ZZ2255
+006306	767000		ZZ2255=ZZ2255+ZZ2255
+006306	756000		ZZ2255=ZZ2255+ZZ2255
+006306	734000		ZZ2255=ZZ2255+ZZ2255
+006306	670000		ZZ2255=ZZ2255+ZZ2255
+006306	017777		0 8192 -ZZ1255
+006307	670000		0 ZZ2255
-	 mark 54, 447 		/89 pegs
+006310	001576		ZZ2256=ZZ2256+ZZ2256
+006310	003374		ZZ2256=ZZ2256+ZZ2256
+006310	006770		ZZ2256=ZZ2256+ZZ2256
+006310	015760		ZZ2256=ZZ2256+ZZ2256
+006310	033740		ZZ2256=ZZ2256+ZZ2256
+006310	067700		ZZ2256=ZZ2256+ZZ2256
+006310	157600		ZZ2256=ZZ2256+ZZ2256
+006310	337400		ZZ2256=ZZ2256+ZZ2256
+006310	017712		0 8192 -ZZ1256
+006311	337400		0 ZZ2256
-	 mark 54, -443 		/7 ceti
+006312	776210		ZZ2257=ZZ2257+ZZ2257
+006312	774420		ZZ2257=ZZ2257+ZZ2257
+006312	771040		ZZ2257=ZZ2257+ZZ2257
+006312	762100		ZZ2257=ZZ2257+ZZ2257
+006312	744200		ZZ2257=ZZ2257+ZZ2257
+006312	710400		ZZ2257=ZZ2257+ZZ2257
+006312	621000		ZZ2257=ZZ2257+ZZ2257
+006312	442000		ZZ2257=ZZ2257+ZZ2257
+006312	017712		0 8192 -ZZ1257
+006313	442000		0 ZZ2257
-	 mark 82, -214 		/8 ceti
+006314	777122		ZZ2258=ZZ2258+ZZ2258
+006314	776244		ZZ2258=ZZ2258+ZZ2258
+006314	774510		ZZ2258=ZZ2258+ZZ2258
+006314	771220		ZZ2258=ZZ2258+ZZ2258
+006314	762440		ZZ2258=ZZ2258+ZZ2258
+006314	745100		ZZ2258=ZZ2258+ZZ2258
+006314	712200		ZZ2258=ZZ2258+ZZ2258
+006314	624400		ZZ2258=ZZ2258+ZZ2258
+006314	017656		0 8192 -ZZ1258
+006315	624400		0 ZZ2258
-	 mark 223, -254 	/17 ceti
+006316	777002		ZZ2259=ZZ2259+ZZ2259
+006316	776004		ZZ2259=ZZ2259+ZZ2259
+006316	774010		ZZ2259=ZZ2259+ZZ2259
+006316	770020		ZZ2259=ZZ2259+ZZ2259
+006316	760040		ZZ2259=ZZ2259+ZZ2259
+006316	740100		ZZ2259=ZZ2259+ZZ2259
+006316	700200		ZZ2259=ZZ2259+ZZ2259
+006316	600400		ZZ2259=ZZ2259+ZZ2259
+006316	017441		0 8192 -ZZ1259
+006317	600400		0 ZZ2259
-	 mark 248, 160 		/63 pisc
+006320	000500		ZZ2260=ZZ2260+ZZ2260
+006320	001200		ZZ2260=ZZ2260+ZZ2260
+006320	002400		ZZ2260=ZZ2260+ZZ2260
+006320	005000		ZZ2260=ZZ2260+ZZ2260
+006320	012000		ZZ2260=ZZ2260+ZZ2260
+006320	024000		ZZ2260=ZZ2260+ZZ2260
+006320	050000		ZZ2260=ZZ2260+ZZ2260
+006320	120000		ZZ2260=ZZ2260+ZZ2260
+006320	017410		0 8192 -ZZ1260
+006321	120000		0 ZZ2260
-	 mark 273, -38 		/20 ceti
+006322	777662		ZZ2261=ZZ2261+ZZ2261
+006322	777544		ZZ2261=ZZ2261+ZZ2261
+006322	777310		ZZ2261=ZZ2261+ZZ2261
+006322	776620		ZZ2261=ZZ2261+ZZ2261
+006322	775440		ZZ2261=ZZ2261+ZZ2261
+006322	773100		ZZ2261=ZZ2261+ZZ2261
+006322	766200		ZZ2261=ZZ2261+ZZ2261
+006322	754400		ZZ2261=ZZ2261+ZZ2261
+006322	017357		0 8192 -ZZ1261
+006323	754400		0 ZZ2261
-	 mark 329, 167 		/71 pisc
+006324	000516		ZZ2262=ZZ2262+ZZ2262
+006324	001234		ZZ2262=ZZ2262+ZZ2262
+006324	002470		ZZ2262=ZZ2262+ZZ2262
+006324	005160		ZZ2262=ZZ2262+ZZ2262
+006324	012340		ZZ2262=ZZ2262+ZZ2262
+006324	024700		ZZ2262=ZZ2262+ZZ2262
+006324	051600		ZZ2262=ZZ2262+ZZ2262
+006324	123400		ZZ2262=ZZ2262+ZZ2262
+006324	017267		0 8192 -ZZ1262
+006325	123400		0 ZZ2262
-	 mark 376, 467 		/84 pisc
+006326	001646		ZZ2263=ZZ2263+ZZ2263
+006326	003514		ZZ2263=ZZ2263+ZZ2263
+006326	007230		ZZ2263=ZZ2263+ZZ2263
+006326	016460		ZZ2263=ZZ2263+ZZ2263
+006326	035140		ZZ2263=ZZ2263+ZZ2263
+006326	072300		ZZ2263=ZZ2263+ZZ2263
+006326	164600		ZZ2263=ZZ2263+ZZ2263
+006326	351400		ZZ2263=ZZ2263+ZZ2263
+006326	017210		0 8192 -ZZ1263
+006327	351400		0 ZZ2263
-	 mark 450, -198 	/45 ceti
+006330	777162		ZZ2264=ZZ2264+ZZ2264
+006330	776344		ZZ2264=ZZ2264+ZZ2264
+006330	774710		ZZ2264=ZZ2264+ZZ2264
+006330	771620		ZZ2264=ZZ2264+ZZ2264
+006330	763440		ZZ2264=ZZ2264+ZZ2264
+006330	747100		ZZ2264=ZZ2264+ZZ2264
+006330	716200		ZZ2264=ZZ2264+ZZ2264
+006330	634400		ZZ2264=ZZ2264+ZZ2264
+006330	017076		0 8192 -ZZ1264
+006331	634400		0 ZZ2264
-	 mark 548, 113 		/106 pisc
+006332	000342		ZZ2265=ZZ2265+ZZ2265
+006332	000704		ZZ2265=ZZ2265+ZZ2265
+006332	001610		ZZ2265=ZZ2265+ZZ2265
+006332	003420		ZZ2265=ZZ2265+ZZ2265
+006332	007040		ZZ2265=ZZ2265+ZZ2265
+006332	016100		ZZ2265=ZZ2265+ZZ2265
+006332	034200		ZZ2265=ZZ2265+ZZ2265
+006332	070400		ZZ2265=ZZ2265+ZZ2265
+006332	016734		0 8192 -ZZ1265
+006333	070400		0 ZZ2265
-	 mark 570, 197          /110 pisc
+006334	000612		ZZ2266=ZZ2266+ZZ2266
+006334	001424		ZZ2266=ZZ2266+ZZ2266
+006334	003050		ZZ2266=ZZ2266+ZZ2266
+006334	006120		ZZ2266=ZZ2266+ZZ2266
+006334	014240		ZZ2266=ZZ2266+ZZ2266
+006334	030500		ZZ2266=ZZ2266+ZZ2266
+006334	061200		ZZ2266=ZZ2266+ZZ2266
+006334	142400		ZZ2266=ZZ2266+ZZ2266
+006334	016706		0 8192 -ZZ1266
+006335	142400		0 ZZ2266
-	 mark 595, -255         /53 ceti
+006336	777000		ZZ2267=ZZ2267+ZZ2267
+006336	776000		ZZ2267=ZZ2267+ZZ2267
+006336	774000		ZZ2267=ZZ2267+ZZ2267
+006336	770000		ZZ2267=ZZ2267+ZZ2267
+006336	760000		ZZ2267=ZZ2267+ZZ2267
+006336	740000		ZZ2267=ZZ2267+ZZ2267
+006336	700000		ZZ2267=ZZ2267+ZZ2267
+006336	600000		ZZ2267=ZZ2267+ZZ2267
+006336	016655		0 8192 -ZZ1267
+006337	600000		0 ZZ2267
-	 mark 606, -247         /55 ceti
+006340	777020		ZZ2268=ZZ2268+ZZ2268
+006340	776040		ZZ2268=ZZ2268+ZZ2268
+006340	774100		ZZ2268=ZZ2268+ZZ2268
+006340	770200		ZZ2268=ZZ2268+ZZ2268
+006340	760400		ZZ2268=ZZ2268+ZZ2268
+006340	741000		ZZ2268=ZZ2268+ZZ2268
+006340	702000		ZZ2268=ZZ2268+ZZ2268
+006340	604000		ZZ2268=ZZ2268+ZZ2268
+006340	016642		0 8192 -ZZ1268
+006341	604000		0 ZZ2268
-	 mark 615, 428          / 5 arie
+006342	001530		ZZ2269=ZZ2269+ZZ2269
+006342	003260		ZZ2269=ZZ2269+ZZ2269
+006342	006540		ZZ2269=ZZ2269+ZZ2269
+006342	015300		ZZ2269=ZZ2269+ZZ2269
+006342	032600		ZZ2269=ZZ2269+ZZ2269
+006342	065400		ZZ2269=ZZ2269+ZZ2269
+006342	153000		ZZ2269=ZZ2269+ZZ2269
+006342	326000		ZZ2269=ZZ2269+ZZ2269
+006342	016631		0 8192 -ZZ1269
+006343	326000		0 ZZ2269
-	 mark 617, 61           /14 pisc
+006344	000172		ZZ2270=ZZ2270+ZZ2270
+006344	000364		ZZ2270=ZZ2270+ZZ2270
+006344	000750		ZZ2270=ZZ2270+ZZ2270
+006344	001720		ZZ2270=ZZ2270+ZZ2270
+006344	003640		ZZ2270=ZZ2270+ZZ2270
+006344	007500		ZZ2270=ZZ2270+ZZ2270
+006344	017200		ZZ2270=ZZ2270+ZZ2270
+006344	036400		ZZ2270=ZZ2270+ZZ2270
+006344	016627		0 8192 -ZZ1270
+006345	036400		0 ZZ2270
-	 mark 656,  -491        /59 ceti
+006346	776050		ZZ2271=ZZ2271+ZZ2271
+006346	774120		ZZ2271=ZZ2271+ZZ2271
+006346	770240		ZZ2271=ZZ2271+ZZ2271
+006346	760500		ZZ2271=ZZ2271+ZZ2271
+006346	741200		ZZ2271=ZZ2271+ZZ2271
+006346	702400		ZZ2271=ZZ2271+ZZ2271
+006346	605000		ZZ2271=ZZ2271+ZZ2271
+006346	412000		ZZ2271=ZZ2271+ZZ2271
+006346	016560		0 8192 -ZZ1271
+006347	412000		0 ZZ2271
-	 mark 665, 52           /113 pisc
+006350	000150		ZZ2272=ZZ2272+ZZ2272
+006350	000320		ZZ2272=ZZ2272+ZZ2272
+006350	000640		ZZ2272=ZZ2272+ZZ2272
+006350	001500		ZZ2272=ZZ2272+ZZ2272
+006350	003200		ZZ2272=ZZ2272+ZZ2272
+006350	006400		ZZ2272=ZZ2272+ZZ2272
+006350	015000		ZZ2272=ZZ2272+ZZ2272
+006350	032000		ZZ2272=ZZ2272+ZZ2272
+006350	016547		0 8192 -ZZ1272
+006351	032000		0 ZZ2272
-	 mark 727, 191          /65 ceti
+006352	000576		ZZ2273=ZZ2273+ZZ2273
+006352	001374		ZZ2273=ZZ2273+ZZ2273
+006352	002770		ZZ2273=ZZ2273+ZZ2273
+006352	005760		ZZ2273=ZZ2273+ZZ2273
+006352	013740		ZZ2273=ZZ2273+ZZ2273
+006352	027700		ZZ2273=ZZ2273+ZZ2273
+006352	057600		ZZ2273=ZZ2273+ZZ2273
+006352	137400		ZZ2273=ZZ2273+ZZ2273
+006352	016451		0 8192 -ZZ1273
+006353	137400		0 ZZ2273
-	 mark 803, -290         /72 ceti
+006354	776672		ZZ2274=ZZ2274+ZZ2274
+006354	775564		ZZ2274=ZZ2274+ZZ2274
+006354	773350		ZZ2274=ZZ2274+ZZ2274
+006354	766720		ZZ2274=ZZ2274+ZZ2274
+006354	755640		ZZ2274=ZZ2274+ZZ2274
+006354	733500		ZZ2274=ZZ2274+ZZ2274
+006354	667200		ZZ2274=ZZ2274+ZZ2274
+006354	556400		ZZ2274=ZZ2274+ZZ2274
+006354	016335		0 8192 -ZZ1274
+006355	556400		0 ZZ2274
-	 mark 813, 182          /73 ceti
+006356	000554		ZZ2275=ZZ2275+ZZ2275
+006356	001330		ZZ2275=ZZ2275+ZZ2275
+006356	002660		ZZ2275=ZZ2275+ZZ2275
+006356	005540		ZZ2275=ZZ2275+ZZ2275
+006356	013300		ZZ2275=ZZ2275+ZZ2275
+006356	026600		ZZ2275=ZZ2275+ZZ2275
+006356	055400		ZZ2275=ZZ2275+ZZ2275
+006356	133000		ZZ2275=ZZ2275+ZZ2275
+006356	016323		0 8192 -ZZ1275
+006357	133000		0 ZZ2275
-	 mark 838, -357         /76 ceti
+006360	776464		ZZ2276=ZZ2276+ZZ2276
+006360	775150		ZZ2276=ZZ2276+ZZ2276
+006360	772320		ZZ2276=ZZ2276+ZZ2276
+006360	764640		ZZ2276=ZZ2276+ZZ2276
+006360	751500		ZZ2276=ZZ2276+ZZ2276
+006360	723200		ZZ2276=ZZ2276+ZZ2276
+006360	646400		ZZ2276=ZZ2276+ZZ2276
+006360	515000		ZZ2276=ZZ2276+ZZ2276
+006360	016272		0 8192 -ZZ1276
+006361	515000		0 ZZ2276
-	 mark 878, -2           /82 ceti
+006362	777772		ZZ2277=ZZ2277+ZZ2277
+006362	777764		ZZ2277=ZZ2277+ZZ2277
+006362	777750		ZZ2277=ZZ2277+ZZ2277
+006362	777720		ZZ2277=ZZ2277+ZZ2277
+006362	777640		ZZ2277=ZZ2277+ZZ2277
+006362	777500		ZZ2277=ZZ2277+ZZ2277
+006362	777200		ZZ2277=ZZ2277+ZZ2277
+006362	776400		ZZ2277=ZZ2277+ZZ2277
+006362	016222		0 8192 -ZZ1277
+006363	776400		0 ZZ2277
-	 mark 907, -340         /89 ceti
+006364	776526		ZZ2278=ZZ2278+ZZ2278
+006364	775254		ZZ2278=ZZ2278+ZZ2278
+006364	772530		ZZ2278=ZZ2278+ZZ2278
+006364	765260		ZZ2278=ZZ2278+ZZ2278
+006364	752540		ZZ2278=ZZ2278+ZZ2278
+006364	725300		ZZ2278=ZZ2278+ZZ2278
+006364	652600		ZZ2278=ZZ2278+ZZ2278
+006364	525400		ZZ2278=ZZ2278+ZZ2278
+006364	016165		0 8192 -ZZ1278
+006365	525400		0 ZZ2278
-	 mark 908, 221          /87 ceti
+006366	000672		ZZ2279=ZZ2279+ZZ2279
+006366	001564		ZZ2279=ZZ2279+ZZ2279
+006366	003350		ZZ2279=ZZ2279+ZZ2279
+006366	006720		ZZ2279=ZZ2279+ZZ2279
+006366	015640		ZZ2279=ZZ2279+ZZ2279
+006366	033500		ZZ2279=ZZ2279+ZZ2279
+006366	067200		ZZ2279=ZZ2279+ZZ2279
+006366	156400		ZZ2279=ZZ2279+ZZ2279
+006366	016164		0 8192 -ZZ1279
+006367	156400		0 ZZ2279
-	 mark 913, -432         / 1 erid
+006370	776236		ZZ2280=ZZ2280+ZZ2280
+006370	774474		ZZ2280=ZZ2280+ZZ2280
+006370	771170		ZZ2280=ZZ2280+ZZ2280
+006370	762360		ZZ2280=ZZ2280+ZZ2280
+006370	744740		ZZ2280=ZZ2280+ZZ2280
+006370	711700		ZZ2280=ZZ2280+ZZ2280
+006370	623600		ZZ2280=ZZ2280+ZZ2280
+006370	447400		ZZ2280=ZZ2280+ZZ2280
+006370	016157		0 8192 -ZZ1280
+006371	447400		0 ZZ2280
-	 mark 947, -487         / 2 erid
+006372	776060		ZZ2281=ZZ2281+ZZ2281
+006372	774140		ZZ2281=ZZ2281+ZZ2281
+006372	770300		ZZ2281=ZZ2281+ZZ2281
+006372	760600		ZZ2281=ZZ2281+ZZ2281
+006372	741400		ZZ2281=ZZ2281+ZZ2281
+006372	703000		ZZ2281=ZZ2281+ZZ2281
+006372	606000		ZZ2281=ZZ2281+ZZ2281
+006372	414000		ZZ2281=ZZ2281+ZZ2281
+006372	016115		0 8192 -ZZ1281
+006373	414000		0 ZZ2281
-	 mark 976, -212         / 3 erid
+006374	777126		ZZ2282=ZZ2282+ZZ2282
+006374	776254		ZZ2282=ZZ2282+ZZ2282
+006374	774530		ZZ2282=ZZ2282+ZZ2282
+006374	771260		ZZ2282=ZZ2282+ZZ2282
+006374	762540		ZZ2282=ZZ2282+ZZ2282
+006374	745300		ZZ2282=ZZ2282+ZZ2282
+006374	712600		ZZ2282=ZZ2282+ZZ2282
+006374	625400		ZZ2282=ZZ2282+ZZ2282
+006374	016060		0 8192 -ZZ1282
+006375	625400		0 ZZ2282
-	 mark 992, 194          /91 ceti
+006376	000604		ZZ2283=ZZ2283+ZZ2283
+006376	001410		ZZ2283=ZZ2283+ZZ2283
+006376	003020		ZZ2283=ZZ2283+ZZ2283
+006376	006040		ZZ2283=ZZ2283+ZZ2283
+006376	014100		ZZ2283=ZZ2283+ZZ2283
+006376	030200		ZZ2283=ZZ2283+ZZ2283
+006376	060400		ZZ2283=ZZ2283+ZZ2283
+006376	141000		ZZ2283=ZZ2283+ZZ2283
+006376	016040		0 8192 -ZZ1283
+006377	141000		0 ZZ2283
-	 mark 1058, 440         /57 arie
+006400	001560		ZZ2284=ZZ2284+ZZ2284
+006400	003340		ZZ2284=ZZ2284+ZZ2284
+006400	006700		ZZ2284=ZZ2284+ZZ2284
+006400	015600		ZZ2284=ZZ2284+ZZ2284
+006400	033400		ZZ2284=ZZ2284+ZZ2284
+006400	067000		ZZ2284=ZZ2284+ZZ2284
+006400	156000		ZZ2284=ZZ2284+ZZ2284
+006400	334000		ZZ2284=ZZ2284+ZZ2284
+006400	015736		0 8192 -ZZ1284
+006401	334000		0 ZZ2284
-	 mark 1076, 470         /58 arie
+006402	001654		ZZ2285=ZZ2285+ZZ2285
+006402	003530		ZZ2285=ZZ2285+ZZ2285
+006402	007260		ZZ2285=ZZ2285+ZZ2285
+006402	016540		ZZ2285=ZZ2285+ZZ2285
+006402	035300		ZZ2285=ZZ2285+ZZ2285
+006402	072600		ZZ2285=ZZ2285+ZZ2285
+006402	165400		ZZ2285=ZZ2285+ZZ2285
+006402	353000		ZZ2285=ZZ2285+ZZ2285
+006402	015714		0 8192 -ZZ1285
+006403	353000		0 ZZ2285
-	 mark 1087,  -209       /13 erid
+006404	777134		ZZ2286=ZZ2286+ZZ2286
+006404	776270		ZZ2286=ZZ2286+ZZ2286
+006404	774560		ZZ2286=ZZ2286+ZZ2286
+006404	771340		ZZ2286=ZZ2286+ZZ2286
+006404	762700		ZZ2286=ZZ2286+ZZ2286
+006404	745600		ZZ2286=ZZ2286+ZZ2286
+006404	713400		ZZ2286=ZZ2286+ZZ2286
+006404	627000		ZZ2286=ZZ2286+ZZ2286
+006404	015701		0 8192 -ZZ1286
+006405	627000		0 ZZ2286
-	 mark 1104, 68          /96 ceti
+006406	000210		ZZ2287=ZZ2287+ZZ2287
+006406	000420		ZZ2287=ZZ2287+ZZ2287
+006406	001040		ZZ2287=ZZ2287+ZZ2287
+006406	002100		ZZ2287=ZZ2287+ZZ2287
+006406	004200		ZZ2287=ZZ2287+ZZ2287
+006406	010400		ZZ2287=ZZ2287+ZZ2287
+006406	021000		ZZ2287=ZZ2287+ZZ2287
+006406	042000		ZZ2287=ZZ2287+ZZ2287
+006406	015660		0 8192 -ZZ1287
+006407	042000		0 ZZ2287
-	 mark 1110, -503        /16 erid
+006410	776020		ZZ2288=ZZ2288+ZZ2288
+006410	774040		ZZ2288=ZZ2288+ZZ2288
+006410	770100		ZZ2288=ZZ2288+ZZ2288
+006410	760200		ZZ2288=ZZ2288+ZZ2288
+006410	740400		ZZ2288=ZZ2288+ZZ2288
+006410	701000		ZZ2288=ZZ2288+ZZ2288
+006410	602000		ZZ2288=ZZ2288+ZZ2288
+006410	404000		ZZ2288=ZZ2288+ZZ2288
+006410	015652		0 8192 -ZZ1288
+006411	404000		0 ZZ2288
-	 mark 1135, 198         / 1 taur
+006412	000614		ZZ2289=ZZ2289+ZZ2289
+006412	001430		ZZ2289=ZZ2289+ZZ2289
+006412	003060		ZZ2289=ZZ2289+ZZ2289
+006412	006140		ZZ2289=ZZ2289+ZZ2289
+006412	014300		ZZ2289=ZZ2289+ZZ2289
+006412	030600		ZZ2289=ZZ2289+ZZ2289
+006412	061400		ZZ2289=ZZ2289+ZZ2289
+006412	143000		ZZ2289=ZZ2289+ZZ2289
+006412	015621		0 8192 -ZZ1289
+006413	143000		0 ZZ2289
-	 mark 1148, 214         / 2 taur
+006414	000654		ZZ2290=ZZ2290+ZZ2290
+006414	001530		ZZ2290=ZZ2290+ZZ2290
+006414	003260		ZZ2290=ZZ2290+ZZ2290
+006414	006540		ZZ2290=ZZ2290+ZZ2290
+006414	015300		ZZ2290=ZZ2290+ZZ2290
+006414	032600		ZZ2290=ZZ2290+ZZ2290
+006414	065400		ZZ2290=ZZ2290+ZZ2290
+006414	153000		ZZ2290=ZZ2290+ZZ2290
+006414	015604		0 8192 -ZZ1290
+006415	153000		0 ZZ2290
-	 mark 1168, 287         / 5 taur
+006416	001076		ZZ2291=ZZ2291+ZZ2291
+006416	002174		ZZ2291=ZZ2291+ZZ2291
+006416	004370		ZZ2291=ZZ2291+ZZ2291
+006416	010760		ZZ2291=ZZ2291+ZZ2291
+006416	021740		ZZ2291=ZZ2291+ZZ2291
+006416	043700		ZZ2291=ZZ2291+ZZ2291
+006416	107600		ZZ2291=ZZ2291+ZZ2291
+006416	217400		ZZ2291=ZZ2291+ZZ2291
+006416	015560		0 8192 -ZZ1291
+006417	217400		0 ZZ2291
-	 mark 1170, -123        /17 erid
+006420	777410		ZZ2292=ZZ2292+ZZ2292
+006420	777020		ZZ2292=ZZ2292+ZZ2292
+006420	776040		ZZ2292=ZZ2292+ZZ2292
+006420	774100		ZZ2292=ZZ2292+ZZ2292
+006420	770200		ZZ2292=ZZ2292+ZZ2292
+006420	760400		ZZ2292=ZZ2292+ZZ2292
+006420	741000		ZZ2292=ZZ2292+ZZ2292
+006420	702000		ZZ2292=ZZ2292+ZZ2292
+006420	015556		0 8192 -ZZ1292
+006421	702000		0 ZZ2292
-	 mark 1185, -223        /18 erid
+006422	777100		ZZ2293=ZZ2293+ZZ2293
+006422	776200		ZZ2293=ZZ2293+ZZ2293
+006422	774400		ZZ2293=ZZ2293+ZZ2293
+006422	771000		ZZ2293=ZZ2293+ZZ2293
+006422	762000		ZZ2293=ZZ2293+ZZ2293
+006422	744000		ZZ2293=ZZ2293+ZZ2293
+006422	710000		ZZ2293=ZZ2293+ZZ2293
+006422	620000		ZZ2293=ZZ2293+ZZ2293
+006422	015537		0 8192 -ZZ1293
+006423	620000		0 ZZ2293
-	 mark 1191, -500        /19 erid
+006424	776026		ZZ2294=ZZ2294+ZZ2294
+006424	774054		ZZ2294=ZZ2294+ZZ2294
+006424	770130		ZZ2294=ZZ2294+ZZ2294
+006424	760260		ZZ2294=ZZ2294+ZZ2294
+006424	740540		ZZ2294=ZZ2294+ZZ2294
+006424	701300		ZZ2294=ZZ2294+ZZ2294
+006424	602600		ZZ2294=ZZ2294+ZZ2294
+006424	405400		ZZ2294=ZZ2294+ZZ2294
+006424	015531		0 8192 -ZZ1294
+006425	405400		0 ZZ2294
-	 mark 1205, 2           /10 taur
+006426	000004		ZZ2295=ZZ2295+ZZ2295
+006426	000010		ZZ2295=ZZ2295+ZZ2295
+006426	000020		ZZ2295=ZZ2295+ZZ2295
+006426	000040		ZZ2295=ZZ2295+ZZ2295
+006426	000100		ZZ2295=ZZ2295+ZZ2295
+006426	000200		ZZ2295=ZZ2295+ZZ2295
+006426	000400		ZZ2295=ZZ2295+ZZ2295
+006426	001000		ZZ2295=ZZ2295+ZZ2295
+006426	015513		0 8192 -ZZ1295
+006427	001000		0 ZZ2295
-	 mark 1260, -283        /26 erid
+006430	776710		ZZ2296=ZZ2296+ZZ2296
+006430	775620		ZZ2296=ZZ2296+ZZ2296
+006430	773440		ZZ2296=ZZ2296+ZZ2296
+006430	767100		ZZ2296=ZZ2296+ZZ2296
+006430	756200		ZZ2296=ZZ2296+ZZ2296
+006430	734400		ZZ2296=ZZ2296+ZZ2296
+006430	671000		ZZ2296=ZZ2296+ZZ2296
+006430	562000		ZZ2296=ZZ2296+ZZ2296
+006430	015424		0 8192 -ZZ1296
+006431	562000		0 ZZ2296
-	 mark 1304, -74         /32 erid
+006432	777552		ZZ2297=ZZ2297+ZZ2297
+006432	777324		ZZ2297=ZZ2297+ZZ2297
+006432	776650		ZZ2297=ZZ2297+ZZ2297
+006432	775520		ZZ2297=ZZ2297+ZZ2297
+006432	773240		ZZ2297=ZZ2297+ZZ2297
+006432	766500		ZZ2297=ZZ2297+ZZ2297
+006432	755200		ZZ2297=ZZ2297+ZZ2297
+006432	732400		ZZ2297=ZZ2297+ZZ2297
+006432	015350		0 8192 -ZZ1297
+006433	732400		0 ZZ2297
-	 mark 1338, 278         /35 taur
+006434	001054		ZZ2298=ZZ2298+ZZ2298
+006434	002130		ZZ2298=ZZ2298+ZZ2298
+006434	004260		ZZ2298=ZZ2298+ZZ2298
+006434	010540		ZZ2298=ZZ2298+ZZ2298
+006434	021300		ZZ2298=ZZ2298+ZZ2298
+006434	042600		ZZ2298=ZZ2298+ZZ2298
+006434	105400		ZZ2298=ZZ2298+ZZ2298
+006434	213000		ZZ2298=ZZ2298+ZZ2298
+006434	015306		0 8192 -ZZ1298
+006435	213000		0 ZZ2298
-	 mark 1353, 130         /38 taur
+006436	000404		ZZ2299=ZZ2299+ZZ2299
+006436	001010		ZZ2299=ZZ2299+ZZ2299
+006436	002020		ZZ2299=ZZ2299+ZZ2299
+006436	004040		ZZ2299=ZZ2299+ZZ2299
+006436	010100		ZZ2299=ZZ2299+ZZ2299
+006436	020200		ZZ2299=ZZ2299+ZZ2299
+006436	040400		ZZ2299=ZZ2299+ZZ2299
+006436	101000		ZZ2299=ZZ2299+ZZ2299
+006436	015267		0 8192 -ZZ1299
+006437	101000		0 ZZ2299
-	 mark 1358, 497         /37 taur
+006440	001742		ZZ2300=ZZ2300+ZZ2300
+006440	003704		ZZ2300=ZZ2300+ZZ2300
+006440	007610		ZZ2300=ZZ2300+ZZ2300
+006440	017420		ZZ2300=ZZ2300+ZZ2300
+006440	037040		ZZ2300=ZZ2300+ZZ2300
+006440	076100		ZZ2300=ZZ2300+ZZ2300
+006440	174200		ZZ2300=ZZ2300+ZZ2300
+006440	370400		ZZ2300=ZZ2300+ZZ2300
+006440	015262		0 8192 -ZZ1300
+006441	370400		0 ZZ2300
-	 mark 1405, -162        /38 erid
+006442	777272		ZZ2301=ZZ2301+ZZ2301
+006442	776564		ZZ2301=ZZ2301+ZZ2301
+006442	775350		ZZ2301=ZZ2301+ZZ2301
+006442	772720		ZZ2301=ZZ2301+ZZ2301
+006442	765640		ZZ2301=ZZ2301+ZZ2301
+006442	753500		ZZ2301=ZZ2301+ZZ2301
+006442	727200		ZZ2301=ZZ2301+ZZ2301
+006442	656400		ZZ2301=ZZ2301+ZZ2301
+006442	015203		0 8192 -ZZ1301
+006443	656400		0 ZZ2301
-	 mark 1414,  205        /47 taur
+006444	000632		ZZ2302=ZZ2302+ZZ2302
+006444	001464		ZZ2302=ZZ2302+ZZ2302
+006444	003150		ZZ2302=ZZ2302+ZZ2302
+006444	006320		ZZ2302=ZZ2302+ZZ2302
+006444	014640		ZZ2302=ZZ2302+ZZ2302
+006444	031500		ZZ2302=ZZ2302+ZZ2302
+006444	063200		ZZ2302=ZZ2302+ZZ2302
+006444	146400		ZZ2302=ZZ2302+ZZ2302
+006444	015172		0 8192 -ZZ1302
+006445	146400		0 ZZ2302
-	 mark 1423, 197         /49 taur
+006446	000612		ZZ2303=ZZ2303+ZZ2303
+006446	001424		ZZ2303=ZZ2303+ZZ2303
+006446	003050		ZZ2303=ZZ2303+ZZ2303
+006446	006120		ZZ2303=ZZ2303+ZZ2303
+006446	014240		ZZ2303=ZZ2303+ZZ2303
+006446	030500		ZZ2303=ZZ2303+ZZ2303
+006446	061200		ZZ2303=ZZ2303+ZZ2303
+006446	142400		ZZ2303=ZZ2303+ZZ2303
+006446	015161		0 8192 -ZZ1303
+006447	142400		0 ZZ2303
-	 mark 1426, -178        /40 erid
+006450	777232		ZZ2304=ZZ2304+ZZ2304
+006450	776464		ZZ2304=ZZ2304+ZZ2304
+006450	775150		ZZ2304=ZZ2304+ZZ2304
+006450	772320		ZZ2304=ZZ2304+ZZ2304
+006450	764640		ZZ2304=ZZ2304+ZZ2304
+006450	751500		ZZ2304=ZZ2304+ZZ2304
+006450	723200		ZZ2304=ZZ2304+ZZ2304
+006450	646400		ZZ2304=ZZ2304+ZZ2304
+006450	015156		0 8192 -ZZ1304
+006451	646400		0 ZZ2304
-	 mark 1430, 463         /50 taur
+006452	001636		ZZ2305=ZZ2305+ZZ2305
+006452	003474		ZZ2305=ZZ2305+ZZ2305
+006452	007170		ZZ2305=ZZ2305+ZZ2305
+006452	016360		ZZ2305=ZZ2305+ZZ2305
+006452	034740		ZZ2305=ZZ2305+ZZ2305
+006452	071700		ZZ2305=ZZ2305+ZZ2305
+006452	163600		ZZ2305=ZZ2305+ZZ2305
+006452	347400		ZZ2305=ZZ2305+ZZ2305
+006452	015152		0 8192 -ZZ1305
+006453	347400		0 ZZ2305
-	 mark 1446, 350         /54 taur
+006454	001274		ZZ2306=ZZ2306+ZZ2306
+006454	002570		ZZ2306=ZZ2306+ZZ2306
+006454	005360		ZZ2306=ZZ2306+ZZ2306
+006454	012740		ZZ2306=ZZ2306+ZZ2306
+006454	025700		ZZ2306=ZZ2306+ZZ2306
+006454	053600		ZZ2306=ZZ2306+ZZ2306
+006454	127400		ZZ2306=ZZ2306+ZZ2306
+006454	257000		ZZ2306=ZZ2306+ZZ2306
+006454	015132		0 8192 -ZZ1306
+006455	257000		0 ZZ2306
-	 mark 1463, 394         /61 taur
+006456	001424		ZZ2307=ZZ2307+ZZ2307
+006456	003050		ZZ2307=ZZ2307+ZZ2307
+006456	006120		ZZ2307=ZZ2307+ZZ2307
+006456	014240		ZZ2307=ZZ2307+ZZ2307
+006456	030500		ZZ2307=ZZ2307+ZZ2307
+006456	061200		ZZ2307=ZZ2307+ZZ2307
+006456	142400		ZZ2307=ZZ2307+ZZ2307
+006456	305000		ZZ2307=ZZ2307+ZZ2307
+006456	015111		0 8192 -ZZ1307
+006457	305000		0 ZZ2307
-	 mark 1470, 392         /64 taur
+006460	001420		ZZ2308=ZZ2308+ZZ2308
+006460	003040		ZZ2308=ZZ2308+ZZ2308
+006460	006100		ZZ2308=ZZ2308+ZZ2308
+006460	014200		ZZ2308=ZZ2308+ZZ2308
+006460	030400		ZZ2308=ZZ2308+ZZ2308
+006460	061000		ZZ2308=ZZ2308+ZZ2308
+006460	142000		ZZ2308=ZZ2308+ZZ2308
+006460	304000		ZZ2308=ZZ2308+ZZ2308
+006460	015102		0 8192 -ZZ1308
+006461	304000		0 ZZ2308
-	 mark 1476, 502         /65 taur
+006462	001754		ZZ2309=ZZ2309+ZZ2309
+006462	003730		ZZ2309=ZZ2309+ZZ2309
+006462	007660		ZZ2309=ZZ2309+ZZ2309
+006462	017540		ZZ2309=ZZ2309+ZZ2309
+006462	037300		ZZ2309=ZZ2309+ZZ2309
+006462	076600		ZZ2309=ZZ2309+ZZ2309
+006462	175400		ZZ2309=ZZ2309+ZZ2309
+006462	373000		ZZ2309=ZZ2309+ZZ2309
+006462	015074		0 8192 -ZZ1309
+006463	373000		0 ZZ2309
-	 mark 1477, 403         /68 taur
+006464	001446		ZZ2310=ZZ2310+ZZ2310
+006464	003114		ZZ2310=ZZ2310+ZZ2310
+006464	006230		ZZ2310=ZZ2310+ZZ2310
+006464	014460		ZZ2310=ZZ2310+ZZ2310
+006464	031140		ZZ2310=ZZ2310+ZZ2310
+006464	062300		ZZ2310=ZZ2310+ZZ2310
+006464	144600		ZZ2310=ZZ2310+ZZ2310
+006464	311400		ZZ2310=ZZ2310+ZZ2310
+006464	015073		0 8192 -ZZ1310
+006465	311400		0 ZZ2310
-	 mark 1483, 350		/71 taur
+006466	001274		ZZ2311=ZZ2311+ZZ2311
+006466	002570		ZZ2311=ZZ2311+ZZ2311
+006466	005360		ZZ2311=ZZ2311+ZZ2311
+006466	012740		ZZ2311=ZZ2311+ZZ2311
+006466	025700		ZZ2311=ZZ2311+ZZ2311
+006466	053600		ZZ2311=ZZ2311+ZZ2311
+006466	127400		ZZ2311=ZZ2311+ZZ2311
+006466	257000		ZZ2311=ZZ2311+ZZ2311
+006466	015065		0 8192 -ZZ1311
+006467	257000		0 ZZ2311
-	 mark 1485, 330		/73 taur
+006470	001224		ZZ2312=ZZ2312+ZZ2312
+006470	002450		ZZ2312=ZZ2312+ZZ2312
+006470	005120		ZZ2312=ZZ2312+ZZ2312
+006470	012240		ZZ2312=ZZ2312+ZZ2312
+006470	024500		ZZ2312=ZZ2312+ZZ2312
+006470	051200		ZZ2312=ZZ2312+ZZ2312
+006470	122400		ZZ2312=ZZ2312+ZZ2312
+006470	245000		ZZ2312=ZZ2312+ZZ2312
+006470	015063		0 8192 -ZZ1312
+006471	245000		0 ZZ2312
-	 mark 1495, 358		/77 taur
+006472	001314		ZZ2313=ZZ2313+ZZ2313
+006472	002630		ZZ2313=ZZ2313+ZZ2313
+006472	005460		ZZ2313=ZZ2313+ZZ2313
+006472	013140		ZZ2313=ZZ2313+ZZ2313
+006472	026300		ZZ2313=ZZ2313+ZZ2313
+006472	054600		ZZ2313=ZZ2313+ZZ2313
+006472	131400		ZZ2313=ZZ2313+ZZ2313
+006472	263000		ZZ2313=ZZ2313+ZZ2313
+006472	015051		0 8192 -ZZ1313
+006473	263000		0 ZZ2313
-	 mark 1507, 364		/
+006474	001330		ZZ2314=ZZ2314+ZZ2314
+006474	002660		ZZ2314=ZZ2314+ZZ2314
+006474	005540		ZZ2314=ZZ2314+ZZ2314
+006474	013300		ZZ2314=ZZ2314+ZZ2314
+006474	026600		ZZ2314=ZZ2314+ZZ2314
+006474	055400		ZZ2314=ZZ2314+ZZ2314
+006474	133000		ZZ2314=ZZ2314+ZZ2314
+006474	266000		ZZ2314=ZZ2314+ZZ2314
+006474	015035		0 8192 -ZZ1314
+006475	266000		0 ZZ2314
-	 mark 1518, -6		/45 erid
+006476	777762		ZZ2315=ZZ2315+ZZ2315
+006476	777744		ZZ2315=ZZ2315+ZZ2315
+006476	777710		ZZ2315=ZZ2315+ZZ2315
+006476	777620		ZZ2315=ZZ2315+ZZ2315
+006476	777440		ZZ2315=ZZ2315+ZZ2315
+006476	777100		ZZ2315=ZZ2315+ZZ2315
+006476	776200		ZZ2315=ZZ2315+ZZ2315
+006476	774400		ZZ2315=ZZ2315+ZZ2315
+006476	015022		0 8192 -ZZ1315
+006477	774400		0 ZZ2315
-	 mark 1526, 333		/86 taur
+006500	001232		ZZ2316=ZZ2316+ZZ2316
+006500	002464		ZZ2316=ZZ2316+ZZ2316
+006500	005150		ZZ2316=ZZ2316+ZZ2316
+006500	012320		ZZ2316=ZZ2316+ZZ2316
+006500	024640		ZZ2316=ZZ2316+ZZ2316
+006500	051500		ZZ2316=ZZ2316+ZZ2316
+006500	123200		ZZ2316=ZZ2316+ZZ2316
+006500	246400		ZZ2316=ZZ2316+ZZ2316
+006500	015012		0 8192 -ZZ1316
+006501	246400		0 ZZ2316
-	 mark 1537, 226		/88 taur
+006502	000704		ZZ2317=ZZ2317+ZZ2317
+006502	001610		ZZ2317=ZZ2317+ZZ2317
+006502	003420		ZZ2317=ZZ2317+ZZ2317
+006502	007040		ZZ2317=ZZ2317+ZZ2317
+006502	016100		ZZ2317=ZZ2317+ZZ2317
+006502	034200		ZZ2317=ZZ2317+ZZ2317
+006502	070400		ZZ2317=ZZ2317+ZZ2317
+006502	161000		ZZ2317=ZZ2317+ZZ2317
+006502	014777		0 8192 -ZZ1317
+006503	161000		0 ZZ2317
-	 mark 1544, -81		/48 erid
+006504	777534		ZZ2318=ZZ2318+ZZ2318
+006504	777270		ZZ2318=ZZ2318+ZZ2318
+006504	776560		ZZ2318=ZZ2318+ZZ2318
+006504	775340		ZZ2318=ZZ2318+ZZ2318
+006504	772700		ZZ2318=ZZ2318+ZZ2318
+006504	765600		ZZ2318=ZZ2318+ZZ2318
+006504	753400		ZZ2318=ZZ2318+ZZ2318
+006504	727000		ZZ2318=ZZ2318+ZZ2318
+006504	014770		0 8192 -ZZ1318
+006505	727000		0 ZZ2318
-	 mark 1551, 280		/90 taur
+006506	001060		ZZ2319=ZZ2319+ZZ2319
+006506	002140		ZZ2319=ZZ2319+ZZ2319
+006506	004300		ZZ2319=ZZ2319+ZZ2319
+006506	010600		ZZ2319=ZZ2319+ZZ2319
+006506	021400		ZZ2319=ZZ2319+ZZ2319
+006506	043000		ZZ2319=ZZ2319+ZZ2319
+006506	106000		ZZ2319=ZZ2319+ZZ2319
+006506	214000		ZZ2319=ZZ2319+ZZ2319
+006506	014761		0 8192 -ZZ1319
+006507	214000		0 ZZ2319
-	 mark 1556, 358		/92 taur
+006510	001314		ZZ2320=ZZ2320+ZZ2320
+006510	002630		ZZ2320=ZZ2320+ZZ2320
+006510	005460		ZZ2320=ZZ2320+ZZ2320
+006510	013140		ZZ2320=ZZ2320+ZZ2320
+006510	026300		ZZ2320=ZZ2320+ZZ2320
+006510	054600		ZZ2320=ZZ2320+ZZ2320
+006510	131400		ZZ2320=ZZ2320+ZZ2320
+006510	263000		ZZ2320=ZZ2320+ZZ2320
+006510	014754		0 8192 -ZZ1320
+006511	263000		0 ZZ2320
-	 mark 1557, -330	/53 erid
+006512	776552		ZZ2321=ZZ2321+ZZ2321
+006512	775324		ZZ2321=ZZ2321+ZZ2321
+006512	772650		ZZ2321=ZZ2321+ZZ2321
+006512	765520		ZZ2321=ZZ2321+ZZ2321
+006512	753240		ZZ2321=ZZ2321+ZZ2321
+006512	726500		ZZ2321=ZZ2321+ZZ2321
+006512	655200		ZZ2321=ZZ2321+ZZ2321
+006512	532400		ZZ2321=ZZ2321+ZZ2321
+006512	014753		0 8192 -ZZ1321
+006513	532400		0 ZZ2321
-	 mark 1571, -452	/54 erid
+006514	776166		ZZ2322=ZZ2322+ZZ2322
+006514	774354		ZZ2322=ZZ2322+ZZ2322
+006514	770730		ZZ2322=ZZ2322+ZZ2322
+006514	761660		ZZ2322=ZZ2322+ZZ2322
+006514	743540		ZZ2322=ZZ2322+ZZ2322
+006514	707300		ZZ2322=ZZ2322+ZZ2322
+006514	616600		ZZ2322=ZZ2322+ZZ2322
+006514	435400		ZZ2322=ZZ2322+ZZ2322
+006514	014735		0 8192 -ZZ1322
+006515	435400		0 ZZ2322
-	 mark 1596, -78		/57 erid
+006516	777542		ZZ2323=ZZ2323+ZZ2323
+006516	777304		ZZ2323=ZZ2323+ZZ2323
+006516	776610		ZZ2323=ZZ2323+ZZ2323
+006516	775420		ZZ2323=ZZ2323+ZZ2323
+006516	773040		ZZ2323=ZZ2323+ZZ2323
+006516	766100		ZZ2323=ZZ2323+ZZ2323
+006516	754200		ZZ2323=ZZ2323+ZZ2323
+006516	730400		ZZ2323=ZZ2323+ZZ2323
+006516	014704		0 8192 -ZZ1323
+006517	730400		0 ZZ2323
-	 mark 1622, 199		/ 2 orio
+006520	000616		ZZ2324=ZZ2324+ZZ2324
+006520	001434		ZZ2324=ZZ2324+ZZ2324
+006520	003070		ZZ2324=ZZ2324+ZZ2324
+006520	006160		ZZ2324=ZZ2324+ZZ2324
+006520	014340		ZZ2324=ZZ2324+ZZ2324
+006520	030700		ZZ2324=ZZ2324+ZZ2324
+006520	061600		ZZ2324=ZZ2324+ZZ2324
+006520	143400		ZZ2324=ZZ2324+ZZ2324
+006520	014652		0 8192 -ZZ1324
+006521	143400		0 ZZ2324
-	 mark 1626, 124		/ 3 orio
+006522	000370		ZZ2325=ZZ2325+ZZ2325
+006522	000760		ZZ2325=ZZ2325+ZZ2325
+006522	001740		ZZ2325=ZZ2325+ZZ2325
+006522	003700		ZZ2325=ZZ2325+ZZ2325
+006522	007600		ZZ2325=ZZ2325+ZZ2325
+006522	017400		ZZ2325=ZZ2325+ZZ2325
+006522	037000		ZZ2325=ZZ2325+ZZ2325
+006522	076000		ZZ2325=ZZ2325+ZZ2325
+006522	014646		0 8192 -ZZ1325
+006523	076000		0 ZZ2325
-	 mark 1638, -128	/61 erid
+006524	777376		ZZ2326=ZZ2326+ZZ2326
+006524	776774		ZZ2326=ZZ2326+ZZ2326
+006524	775770		ZZ2326=ZZ2326+ZZ2326
+006524	773760		ZZ2326=ZZ2326+ZZ2326
+006524	767740		ZZ2326=ZZ2326+ZZ2326
+006524	757700		ZZ2326=ZZ2326+ZZ2326
+006524	737600		ZZ2326=ZZ2326+ZZ2326
+006524	677400		ZZ2326=ZZ2326+ZZ2326
+006524	014632		0 8192 -ZZ1326
+006525	677400		0 ZZ2326
-	 mark 1646, 228		/ 7 orio
+006526	000710		ZZ2327=ZZ2327+ZZ2327
+006526	001620		ZZ2327=ZZ2327+ZZ2327
+006526	003440		ZZ2327=ZZ2327+ZZ2327
+006526	007100		ZZ2327=ZZ2327+ZZ2327
+006526	016200		ZZ2327=ZZ2327+ZZ2327
+006526	034400		ZZ2327=ZZ2327+ZZ2327
+006526	071000		ZZ2327=ZZ2327+ZZ2327
+006526	162000		ZZ2327=ZZ2327+ZZ2327
+006526	014622		0 8192 -ZZ1327
+006527	162000		0 ZZ2327
-	 mark 1654, 304		/ 9 orio
+006530	001140		ZZ2328=ZZ2328+ZZ2328
+006530	002300		ZZ2328=ZZ2328+ZZ2328
+006530	004600		ZZ2328=ZZ2328+ZZ2328
+006530	011400		ZZ2328=ZZ2328+ZZ2328
+006530	023000		ZZ2328=ZZ2328+ZZ2328
+006530	046000		ZZ2328=ZZ2328+ZZ2328
+006530	114000		ZZ2328=ZZ2328+ZZ2328
+006530	230000		ZZ2328=ZZ2328+ZZ2328
+006530	014612		0 8192 -ZZ1328
+006531	230000		0 ZZ2328
-	 mark 1669, 36		/10 orio
+006532	000110		ZZ2329=ZZ2329+ZZ2329
+006532	000220		ZZ2329=ZZ2329+ZZ2329
+006532	000440		ZZ2329=ZZ2329+ZZ2329
+006532	001100		ZZ2329=ZZ2329+ZZ2329
+006532	002200		ZZ2329=ZZ2329+ZZ2329
+006532	004400		ZZ2329=ZZ2329+ZZ2329
+006532	011000		ZZ2329=ZZ2329+ZZ2329
+006532	022000		ZZ2329=ZZ2329+ZZ2329
+006532	014573		0 8192 -ZZ1329
+006533	022000		0 ZZ2329
-	 mark 1680, -289	/64 erid
+006534	776674		ZZ2330=ZZ2330+ZZ2330
+006534	775570		ZZ2330=ZZ2330+ZZ2330
+006534	773360		ZZ2330=ZZ2330+ZZ2330
+006534	766740		ZZ2330=ZZ2330+ZZ2330
+006534	755700		ZZ2330=ZZ2330+ZZ2330
+006534	733600		ZZ2330=ZZ2330+ZZ2330
+006534	667400		ZZ2330=ZZ2330+ZZ2330
+006534	557000		ZZ2330=ZZ2330+ZZ2330
+006534	014560		0 8192 -ZZ1330
+006535	557000		0 ZZ2330
-	 mark 1687, -167	/65 erid
+006536	777260		ZZ2331=ZZ2331+ZZ2331
+006536	776540		ZZ2331=ZZ2331+ZZ2331
+006536	775300		ZZ2331=ZZ2331+ZZ2331
+006536	772600		ZZ2331=ZZ2331+ZZ2331
+006536	765400		ZZ2331=ZZ2331+ZZ2331
+006536	753000		ZZ2331=ZZ2331+ZZ2331
+006536	726000		ZZ2331=ZZ2331+ZZ2331
+006536	654000		ZZ2331=ZZ2331+ZZ2331
+006536	014551		0 8192 -ZZ1331
+006537	654000		0 ZZ2331
-	 mark 1690, -460	/
+006540	776146		ZZ2332=ZZ2332+ZZ2332
+006540	774314		ZZ2332=ZZ2332+ZZ2332
+006540	770630		ZZ2332=ZZ2332+ZZ2332
+006540	761460		ZZ2332=ZZ2332+ZZ2332
+006540	743140		ZZ2332=ZZ2332+ZZ2332
+006540	706300		ZZ2332=ZZ2332+ZZ2332
+006540	614600		ZZ2332=ZZ2332+ZZ2332
+006540	431400		ZZ2332=ZZ2332+ZZ2332
+006540	014546		0 8192 -ZZ1332
+006541	431400		0 ZZ2332
-	 mark 1690, 488		/102 taur
+006542	001720		ZZ2333=ZZ2333+ZZ2333
+006542	003640		ZZ2333=ZZ2333+ZZ2333
+006542	007500		ZZ2333=ZZ2333+ZZ2333
+006542	017200		ZZ2333=ZZ2333+ZZ2333
+006542	036400		ZZ2333=ZZ2333+ZZ2333
+006542	075000		ZZ2333=ZZ2333+ZZ2333
+006542	172000		ZZ2333=ZZ2333+ZZ2333
+006542	364000		ZZ2333=ZZ2333+ZZ2333
+006542	014546		0 8192 -ZZ1333
+006543	364000		0 ZZ2333
-	 mark 1700, 347		/11 orio
+006544	001266		ZZ2334=ZZ2334+ZZ2334
+006544	002554		ZZ2334=ZZ2334+ZZ2334
+006544	005330		ZZ2334=ZZ2334+ZZ2334
+006544	012660		ZZ2334=ZZ2334+ZZ2334
+006544	025540		ZZ2334=ZZ2334+ZZ2334
+006544	053300		ZZ2334=ZZ2334+ZZ2334
+006544	126600		ZZ2334=ZZ2334+ZZ2334
+006544	255400		ZZ2334=ZZ2334+ZZ2334
+006544	014534		0 8192 -ZZ1334
+006545	255400		0 ZZ2334
-	 mark 1729, 352		/15 orio
+006546	001300		ZZ2335=ZZ2335+ZZ2335
+006546	002600		ZZ2335=ZZ2335+ZZ2335
+006546	005400		ZZ2335=ZZ2335+ZZ2335
+006546	013000		ZZ2335=ZZ2335+ZZ2335
+006546	026000		ZZ2335=ZZ2335+ZZ2335
+006546	054000		ZZ2335=ZZ2335+ZZ2335
+006546	130000		ZZ2335=ZZ2335+ZZ2335
+006546	260000		ZZ2335=ZZ2335+ZZ2335
+006546	014477		0 8192 -ZZ1335
+006547	260000		0 ZZ2335
-	 mark 1732, -202	/69 erid
+006550	777152		ZZ2336=ZZ2336+ZZ2336
+006550	776324		ZZ2336=ZZ2336+ZZ2336
+006550	774650		ZZ2336=ZZ2336+ZZ2336
+006550	771520		ZZ2336=ZZ2336+ZZ2336
+006550	763240		ZZ2336=ZZ2336+ZZ2336
+006550	746500		ZZ2336=ZZ2336+ZZ2336
+006550	715200		ZZ2336=ZZ2336+ZZ2336
+006550	632400		ZZ2336=ZZ2336+ZZ2336
+006550	014474		0 8192 -ZZ1336
+006551	632400		0 ZZ2336
-	 mark 1750, -273	/ 3 leps
+006552	776734		ZZ2337=ZZ2337+ZZ2337
+006552	775670		ZZ2337=ZZ2337+ZZ2337
+006552	773560		ZZ2337=ZZ2337+ZZ2337
+006552	767340		ZZ2337=ZZ2337+ZZ2337
+006552	756700		ZZ2337=ZZ2337+ZZ2337
+006552	735600		ZZ2337=ZZ2337+ZZ2337
+006552	673400		ZZ2337=ZZ2337+ZZ2337
+006552	567000		ZZ2337=ZZ2337+ZZ2337
+006552	014452		0 8192 -ZZ1337
+006553	567000		0 ZZ2337
-	 mark 1753, 63		/17 orio
+006554	000176		ZZ2338=ZZ2338+ZZ2338
+006554	000374		ZZ2338=ZZ2338+ZZ2338
+006554	000770		ZZ2338=ZZ2338+ZZ2338
+006554	001760		ZZ2338=ZZ2338+ZZ2338
+006554	003740		ZZ2338=ZZ2338+ZZ2338
+006554	007700		ZZ2338=ZZ2338+ZZ2338
+006554	017600		ZZ2338=ZZ2338+ZZ2338
+006554	037400		ZZ2338=ZZ2338+ZZ2338
+006554	014447		0 8192 -ZZ1338
+006555	037400		0 ZZ2338
-	 mark 1756, -297	/ 4 leps
+006556	776654		ZZ2339=ZZ2339+ZZ2339
+006556	775530		ZZ2339=ZZ2339+ZZ2339
+006556	773260		ZZ2339=ZZ2339+ZZ2339
+006556	766540		ZZ2339=ZZ2339+ZZ2339
+006556	755300		ZZ2339=ZZ2339+ZZ2339
+006556	732600		ZZ2339=ZZ2339+ZZ2339
+006556	665400		ZZ2339=ZZ2339+ZZ2339
+006556	553000		ZZ2339=ZZ2339+ZZ2339
+006556	014444		0 8192 -ZZ1339
+006557	553000		0 ZZ2339
-	 mark 1792, -302	/ 6 leps
+006560	776642		ZZ2340=ZZ2340+ZZ2340
+006560	775504		ZZ2340=ZZ2340+ZZ2340
+006560	773210		ZZ2340=ZZ2340+ZZ2340
+006560	766420		ZZ2340=ZZ2340+ZZ2340
+006560	755040		ZZ2340=ZZ2340+ZZ2340
+006560	732100		ZZ2340=ZZ2340+ZZ2340
+006560	664200		ZZ2340=ZZ2340+ZZ2340
+006560	550400		ZZ2340=ZZ2340+ZZ2340
+006560	014400		0 8192 -ZZ1340
+006561	550400		0 ZZ2340
-	 mark 1799, -486	/
+006562	776062		ZZ2341=ZZ2341+ZZ2341
+006562	774144		ZZ2341=ZZ2341+ZZ2341
+006562	770310		ZZ2341=ZZ2341+ZZ2341
+006562	760620		ZZ2341=ZZ2341+ZZ2341
+006562	741440		ZZ2341=ZZ2341+ZZ2341
+006562	703100		ZZ2341=ZZ2341+ZZ2341
+006562	606200		ZZ2341=ZZ2341+ZZ2341
+006562	414400		ZZ2341=ZZ2341+ZZ2341
+006562	014371		0 8192 -ZZ1341
+006563	414400		0 ZZ2341
-	 mark 1801, -11		/22 orio
+006564	777750		ZZ2342=ZZ2342+ZZ2342
+006564	777720		ZZ2342=ZZ2342+ZZ2342
+006564	777640		ZZ2342=ZZ2342+ZZ2342
+006564	777500		ZZ2342=ZZ2342+ZZ2342
+006564	777200		ZZ2342=ZZ2342+ZZ2342
+006564	776400		ZZ2342=ZZ2342+ZZ2342
+006564	775000		ZZ2342=ZZ2342+ZZ2342
+006564	772000		ZZ2342=ZZ2342+ZZ2342
+006564	014367		0 8192 -ZZ1342
+006565	772000		0 ZZ2342
-	 mark 1807, 79		/23 orio
+006566	000236		ZZ2343=ZZ2343+ZZ2343
+006566	000474		ZZ2343=ZZ2343+ZZ2343
+006566	001170		ZZ2343=ZZ2343+ZZ2343
+006566	002360		ZZ2343=ZZ2343+ZZ2343
+006566	004740		ZZ2343=ZZ2343+ZZ2343
+006566	011700		ZZ2343=ZZ2343+ZZ2343
+006566	023600		ZZ2343=ZZ2343+ZZ2343
+006566	047400		ZZ2343=ZZ2343+ZZ2343
+006566	014361		0 8192 -ZZ1343
+006567	047400		0 ZZ2343
-	 mark 1816, -180	/29 orio
+006570	777226		ZZ2344=ZZ2344+ZZ2344
+006570	776454		ZZ2344=ZZ2344+ZZ2344
+006570	775130		ZZ2344=ZZ2344+ZZ2344
+006570	772260		ZZ2344=ZZ2344+ZZ2344
+006570	764540		ZZ2344=ZZ2344+ZZ2344
+006570	751300		ZZ2344=ZZ2344+ZZ2344
+006570	722600		ZZ2344=ZZ2344+ZZ2344
+006570	645400		ZZ2344=ZZ2344+ZZ2344
+006570	014350		0 8192 -ZZ1344
+006571	645400		0 ZZ2344
-	 mark 1818, 40		/25 orio
+006572	000120		ZZ2345=ZZ2345+ZZ2345
+006572	000240		ZZ2345=ZZ2345+ZZ2345
+006572	000500		ZZ2345=ZZ2345+ZZ2345
+006572	001200		ZZ2345=ZZ2345+ZZ2345
+006572	002400		ZZ2345=ZZ2345+ZZ2345
+006572	005000		ZZ2345=ZZ2345+ZZ2345
+006572	012000		ZZ2345=ZZ2345+ZZ2345
+006572	024000		ZZ2345=ZZ2345+ZZ2345
+006572	014346		0 8192 -ZZ1345
+006573	024000		0 ZZ2345
-	 mark 1830, 497		/114 taur
+006574	001742		ZZ2346=ZZ2346+ZZ2346
+006574	003704		ZZ2346=ZZ2346+ZZ2346
+006574	007610		ZZ2346=ZZ2346+ZZ2346
+006574	017420		ZZ2346=ZZ2346+ZZ2346
+006574	037040		ZZ2346=ZZ2346+ZZ2346
+006574	076100		ZZ2346=ZZ2346+ZZ2346
+006574	174200		ZZ2346=ZZ2346+ZZ2346
+006574	370400		ZZ2346=ZZ2346+ZZ2346
+006574	014332		0 8192 -ZZ1346
+006575	370400		0 ZZ2346
-	 mark 1830, 69		/30 orio
+006576	000212		ZZ2347=ZZ2347+ZZ2347
+006576	000424		ZZ2347=ZZ2347+ZZ2347
+006576	001050		ZZ2347=ZZ2347+ZZ2347
+006576	002120		ZZ2347=ZZ2347+ZZ2347
+006576	004240		ZZ2347=ZZ2347+ZZ2347
+006576	010500		ZZ2347=ZZ2347+ZZ2347
+006576	021200		ZZ2347=ZZ2347+ZZ2347
+006576	042400		ZZ2347=ZZ2347+ZZ2347
+006576	014332		0 8192 -ZZ1347
+006577	042400		0 ZZ2347
-	 mark 1851, 134		/32 orio
+006600	000414		ZZ2348=ZZ2348+ZZ2348
+006600	001030		ZZ2348=ZZ2348+ZZ2348
+006600	002060		ZZ2348=ZZ2348+ZZ2348
+006600	004140		ZZ2348=ZZ2348+ZZ2348
+006600	010300		ZZ2348=ZZ2348+ZZ2348
+006600	020600		ZZ2348=ZZ2348+ZZ2348
+006600	041400		ZZ2348=ZZ2348+ZZ2348
+006600	103000		ZZ2348=ZZ2348+ZZ2348
+006600	014305		0 8192 -ZZ1348
+006601	103000		0 ZZ2348
-	 mark 1857, 421		/119 taur
+006602	001512		ZZ2349=ZZ2349+ZZ2349
+006602	003224		ZZ2349=ZZ2349+ZZ2349
+006602	006450		ZZ2349=ZZ2349+ZZ2349
+006602	015120		ZZ2349=ZZ2349+ZZ2349
+006602	032240		ZZ2349=ZZ2349+ZZ2349
+006602	064500		ZZ2349=ZZ2349+ZZ2349
+006602	151200		ZZ2349=ZZ2349+ZZ2349
+006602	322400		ZZ2349=ZZ2349+ZZ2349
+006602	014277		0 8192 -ZZ1349
+006603	322400		0 ZZ2349
-	 mark 1861, -168	/36 orio
+006604	777256		ZZ2350=ZZ2350+ZZ2350
+006604	776534		ZZ2350=ZZ2350+ZZ2350
+006604	775270		ZZ2350=ZZ2350+ZZ2350
+006604	772560		ZZ2350=ZZ2350+ZZ2350
+006604	765340		ZZ2350=ZZ2350+ZZ2350
+006604	752700		ZZ2350=ZZ2350+ZZ2350
+006604	725600		ZZ2350=ZZ2350+ZZ2350
+006604	653400		ZZ2350=ZZ2350+ZZ2350
+006604	014273		0 8192 -ZZ1350
+006605	653400		0 ZZ2350
-	 mark 1874, 214		/37 orio
+006606	000654		ZZ2351=ZZ2351+ZZ2351
+006606	001530		ZZ2351=ZZ2351+ZZ2351
+006606	003260		ZZ2351=ZZ2351+ZZ2351
+006606	006540		ZZ2351=ZZ2351+ZZ2351
+006606	015300		ZZ2351=ZZ2351+ZZ2351
+006606	032600		ZZ2351=ZZ2351+ZZ2351
+006606	065400		ZZ2351=ZZ2351+ZZ2351
+006606	153000		ZZ2351=ZZ2351+ZZ2351
+006606	014256		0 8192 -ZZ1351
+006607	153000		0 ZZ2351
-	 mark 1878, -132	/
+006610	777366		ZZ2352=ZZ2352+ZZ2352
+006610	776754		ZZ2352=ZZ2352+ZZ2352
+006610	775730		ZZ2352=ZZ2352+ZZ2352
+006610	773660		ZZ2352=ZZ2352+ZZ2352
+006610	767540		ZZ2352=ZZ2352+ZZ2352
+006610	757300		ZZ2352=ZZ2352+ZZ2352
+006610	736600		ZZ2352=ZZ2352+ZZ2352
+006610	675400		ZZ2352=ZZ2352+ZZ2352
+006610	014252		0 8192 -ZZ1352
+006611	675400		0 ZZ2352
-	 mark 1880, -112	/42 orio
+006612	777436		ZZ2353=ZZ2353+ZZ2353
+006612	777074		ZZ2353=ZZ2353+ZZ2353
+006612	776170		ZZ2353=ZZ2353+ZZ2353
+006612	774360		ZZ2353=ZZ2353+ZZ2353
+006612	770740		ZZ2353=ZZ2353+ZZ2353
+006612	761700		ZZ2353=ZZ2353+ZZ2353
+006612	743600		ZZ2353=ZZ2353+ZZ2353
+006612	707400		ZZ2353=ZZ2353+ZZ2353
+006612	014250		0 8192 -ZZ1353
+006613	707400		0 ZZ2353
-	 mark 1885, 210		/40 orio
+006614	000644		ZZ2354=ZZ2354+ZZ2354
+006614	001510		ZZ2354=ZZ2354+ZZ2354
+006614	003220		ZZ2354=ZZ2354+ZZ2354
+006614	006440		ZZ2354=ZZ2354+ZZ2354
+006614	015100		ZZ2354=ZZ2354+ZZ2354
+006614	032200		ZZ2354=ZZ2354+ZZ2354
+006614	064400		ZZ2354=ZZ2354+ZZ2354
+006614	151000		ZZ2354=ZZ2354+ZZ2354
+006614	014243		0 8192 -ZZ1354
+006615	151000		0 ZZ2354
-	 mark 1899,-60		/48 orio
+006616	777606		ZZ2355=ZZ2355+ZZ2355
+006616	777414		ZZ2355=ZZ2355+ZZ2355
+006616	777030		ZZ2355=ZZ2355+ZZ2355
+006616	776060		ZZ2355=ZZ2355+ZZ2355
+006616	774140		ZZ2355=ZZ2355+ZZ2355
+006616	770300		ZZ2355=ZZ2355+ZZ2355
+006616	760600		ZZ2355=ZZ2355+ZZ2355
+006616	741400		ZZ2355=ZZ2355+ZZ2355
+006616	014225		0 8192 -ZZ1355
+006617	741400		0 ZZ2355
-	 mark 1900, 93		/47 orio
+006620	000272		ZZ2356=ZZ2356+ZZ2356
+006620	000564		ZZ2356=ZZ2356+ZZ2356
+006620	001350		ZZ2356=ZZ2356+ZZ2356
+006620	002720		ZZ2356=ZZ2356+ZZ2356
+006620	005640		ZZ2356=ZZ2356+ZZ2356
+006620	013500		ZZ2356=ZZ2356+ZZ2356
+006620	027200		ZZ2356=ZZ2356+ZZ2356
+006620	056400		ZZ2356=ZZ2356+ZZ2356
+006620	014224		0 8192 -ZZ1356
+006621	056400		0 ZZ2356
-	 mark 1900, -165	/49 orio
+006622	777264		ZZ2357=ZZ2357+ZZ2357
+006622	776550		ZZ2357=ZZ2357+ZZ2357
+006622	775320		ZZ2357=ZZ2357+ZZ2357
+006622	772640		ZZ2357=ZZ2357+ZZ2357
+006622	765500		ZZ2357=ZZ2357+ZZ2357
+006622	753200		ZZ2357=ZZ2357+ZZ2357
+006622	726400		ZZ2357=ZZ2357+ZZ2357
+006622	655000		ZZ2357=ZZ2357+ZZ2357
+006622	014224		0 8192 -ZZ1357
+006623	655000		0 ZZ2357
-	 mark 1909, 375		/126 taur
+006624	001356		ZZ2358=ZZ2358+ZZ2358
+006624	002734		ZZ2358=ZZ2358+ZZ2358
+006624	005670		ZZ2358=ZZ2358+ZZ2358
+006624	013560		ZZ2358=ZZ2358+ZZ2358
+006624	027340		ZZ2358=ZZ2358+ZZ2358
+006624	056700		ZZ2358=ZZ2358+ZZ2358
+006624	135600		ZZ2358=ZZ2358+ZZ2358
+006624	273400		ZZ2358=ZZ2358+ZZ2358
+006624	014213		0 8192 -ZZ1358
+006625	273400		0 ZZ2358
-	 mark 1936, -511	/13 leps
+006626	776000		ZZ2359=ZZ2359+ZZ2359
+006626	774000		ZZ2359=ZZ2359+ZZ2359
+006626	770000		ZZ2359=ZZ2359+ZZ2359
+006626	760000		ZZ2359=ZZ2359+ZZ2359
+006626	740000		ZZ2359=ZZ2359+ZZ2359
+006626	700000		ZZ2359=ZZ2359+ZZ2359
+006626	600000		ZZ2359=ZZ2359+ZZ2359
+006626	400000		ZZ2359=ZZ2359+ZZ2359
+006626	014160		0 8192 -ZZ1359
+006627	400000		0 ZZ2359
-	 mark 1957, 287		/134 taur
+006630	001076		ZZ2360=ZZ2360+ZZ2360
+006630	002174		ZZ2360=ZZ2360+ZZ2360
+006630	004370		ZZ2360=ZZ2360+ZZ2360
+006630	010760		ZZ2360=ZZ2360+ZZ2360
+006630	021740		ZZ2360=ZZ2360+ZZ2360
+006630	043700		ZZ2360=ZZ2360+ZZ2360
+006630	107600		ZZ2360=ZZ2360+ZZ2360
+006630	217400		ZZ2360=ZZ2360+ZZ2360
+006630	014133		0 8192 -ZZ1360
+006631	217400		0 ZZ2360
-	 mark 1974, -475	/15 leps
+006632	776110		ZZ2361=ZZ2361+ZZ2361
+006632	774220		ZZ2361=ZZ2361+ZZ2361
+006632	770440		ZZ2361=ZZ2361+ZZ2361
+006632	761100		ZZ2361=ZZ2361+ZZ2361
+006632	742200		ZZ2361=ZZ2361+ZZ2361
+006632	704400		ZZ2361=ZZ2361+ZZ2361
+006632	611000		ZZ2361=ZZ2361+ZZ2361
+006632	422000		ZZ2361=ZZ2361+ZZ2361
+006632	014112		0 8192 -ZZ1361
+006633	422000		0 ZZ2361
-	 mark 1982, 461		/54 orio
+006634	001632		ZZ2362=ZZ2362+ZZ2362
+006634	003464		ZZ2362=ZZ2362+ZZ2362
+006634	007150		ZZ2362=ZZ2362+ZZ2362
+006634	016320		ZZ2362=ZZ2362+ZZ2362
+006634	034640		ZZ2362=ZZ2362+ZZ2362
+006634	071500		ZZ2362=ZZ2362+ZZ2362
+006634	163200		ZZ2362=ZZ2362+ZZ2362
+006634	346400		ZZ2362=ZZ2362+ZZ2362
+006634	014102		0 8192 -ZZ1362
+006635	346400		0 ZZ2362
-	 mark 2002, -323	/16 leps
+006636	776570		ZZ2363=ZZ2363+ZZ2363
+006636	775360		ZZ2363=ZZ2363+ZZ2363
+006636	772740		ZZ2363=ZZ2363+ZZ2363
+006636	765700		ZZ2363=ZZ2363+ZZ2363
+006636	753600		ZZ2363=ZZ2363+ZZ2363
+006636	727400		ZZ2363=ZZ2363+ZZ2363
+006636	657000		ZZ2363=ZZ2363+ZZ2363
+006636	536000		ZZ2363=ZZ2363+ZZ2363
+006636	014056		0 8192 -ZZ1363
+006637	536000		0 ZZ2363
-	 mark 2020, -70		/
+006640	777562		ZZ2364=ZZ2364+ZZ2364
+006640	777344		ZZ2364=ZZ2364+ZZ2364
+006640	776710		ZZ2364=ZZ2364+ZZ2364
+006640	775620		ZZ2364=ZZ2364+ZZ2364
+006640	773440		ZZ2364=ZZ2364+ZZ2364
+006640	767100		ZZ2364=ZZ2364+ZZ2364
+006640	756200		ZZ2364=ZZ2364+ZZ2364
+006640	734400		ZZ2364=ZZ2364+ZZ2364
+006640	014034		0 8192 -ZZ1364
+006641	734400		0 ZZ2364
-	 mark 2030, 220		/61 orio
+006642	000670		ZZ2365=ZZ2365+ZZ2365
+006642	001560		ZZ2365=ZZ2365+ZZ2365
+006642	003340		ZZ2365=ZZ2365+ZZ2365
+006642	006700		ZZ2365=ZZ2365+ZZ2365
+006642	015600		ZZ2365=ZZ2365+ZZ2365
+006642	033400		ZZ2365=ZZ2365+ZZ2365
+006642	067000		ZZ2365=ZZ2365+ZZ2365
+006642	156000		ZZ2365=ZZ2365+ZZ2365
+006642	014022		0 8192 -ZZ1365
+006643	156000		0 ZZ2365
-	 mark 2032, -241	/ 3 mono
+006644	777034		ZZ2366=ZZ2366+ZZ2366
+006644	776070		ZZ2366=ZZ2366+ZZ2366
+006644	774160		ZZ2366=ZZ2366+ZZ2366
+006644	770340		ZZ2366=ZZ2366+ZZ2366
+006644	760700		ZZ2366=ZZ2366+ZZ2366
+006644	741600		ZZ2366=ZZ2366+ZZ2366
+006644	703400		ZZ2366=ZZ2366+ZZ2366
+006644	607000		ZZ2366=ZZ2366+ZZ2366
+006644	014020		0 8192 -ZZ1366
+006645	607000		0 ZZ2366
-	 mark 2037, 458		/62 orio
+006646	001624		ZZ2367=ZZ2367+ZZ2367
+006646	003450		ZZ2367=ZZ2367+ZZ2367
+006646	007120		ZZ2367=ZZ2367+ZZ2367
+006646	016240		ZZ2367=ZZ2367+ZZ2367
+006646	034500		ZZ2367=ZZ2367+ZZ2367
+006646	071200		ZZ2367=ZZ2367+ZZ2367
+006646	162400		ZZ2367=ZZ2367+ZZ2367
+006646	345000		ZZ2367=ZZ2367+ZZ2367
+006646	014013		0 8192 -ZZ1367
+006647	345000		0 ZZ2367
-	 mark 2057, -340	/18 leps
+006650	776526		ZZ2368=ZZ2368+ZZ2368
+006650	775254		ZZ2368=ZZ2368+ZZ2368
+006650	772530		ZZ2368=ZZ2368+ZZ2368
+006650	765260		ZZ2368=ZZ2368+ZZ2368
+006650	752540		ZZ2368=ZZ2368+ZZ2368
+006650	725300		ZZ2368=ZZ2368+ZZ2368
+006650	652600		ZZ2368=ZZ2368+ZZ2368
+006650	525400		ZZ2368=ZZ2368+ZZ2368
+006650	013767		0 8192 -ZZ1368
+006651	525400		0 ZZ2368
-	 mark 2059, 336		/67 orio
+006652	001240		ZZ2369=ZZ2369+ZZ2369
+006652	002500		ZZ2369=ZZ2369+ZZ2369
+006652	005200		ZZ2369=ZZ2369+ZZ2369
+006652	012400		ZZ2369=ZZ2369+ZZ2369
+006652	025000		ZZ2369=ZZ2369+ZZ2369
+006652	052000		ZZ2369=ZZ2369+ZZ2369
+006652	124000		ZZ2369=ZZ2369+ZZ2369
+006652	250000		ZZ2369=ZZ2369+ZZ2369
+006652	013765		0 8192 -ZZ1369
+006653	250000		0 ZZ2369
-	 mark 2084, 368		/69 orio
+006654	001340		ZZ2370=ZZ2370+ZZ2370
+006654	002700		ZZ2370=ZZ2370+ZZ2370
+006654	005600		ZZ2370=ZZ2370+ZZ2370
+006654	013400		ZZ2370=ZZ2370+ZZ2370
+006654	027000		ZZ2370=ZZ2370+ZZ2370
+006654	056000		ZZ2370=ZZ2370+ZZ2370
+006654	134000		ZZ2370=ZZ2370+ZZ2370
+006654	270000		ZZ2370=ZZ2370+ZZ2370
+006654	013734		0 8192 -ZZ1370
+006655	270000		0 ZZ2370
-	 mark 2084, 324		/70 orio
+006656	001210		ZZ2371=ZZ2371+ZZ2371
+006656	002420		ZZ2371=ZZ2371+ZZ2371
+006656	005040		ZZ2371=ZZ2371+ZZ2371
+006656	012100		ZZ2371=ZZ2371+ZZ2371
+006656	024200		ZZ2371=ZZ2371+ZZ2371
+006656	050400		ZZ2371=ZZ2371+ZZ2371
+006656	121000		ZZ2371=ZZ2371+ZZ2371
+006656	242000		ZZ2371=ZZ2371+ZZ2371
+006656	013734		0 8192 -ZZ1371
+006657	242000		0 ZZ2371
-	 mark 2105, -142	/ 5 mono
+006660	777342		ZZ2372=ZZ2372+ZZ2372
+006660	776704		ZZ2372=ZZ2372+ZZ2372
+006660	775610		ZZ2372=ZZ2372+ZZ2372
+006660	773420		ZZ2372=ZZ2372+ZZ2372
+006660	767040		ZZ2372=ZZ2372+ZZ2372
+006660	756100		ZZ2372=ZZ2372+ZZ2372
+006660	734200		ZZ2372=ZZ2372+ZZ2372
+006660	670400		ZZ2372=ZZ2372+ZZ2372
+006660	013707		0 8192 -ZZ1372
+006661	670400		0 ZZ2372
-	 mark 2112, -311	/
+006662	776620		ZZ2373=ZZ2373+ZZ2373
+006662	775440		ZZ2373=ZZ2373+ZZ2373
+006662	773100		ZZ2373=ZZ2373+ZZ2373
+006662	766200		ZZ2373=ZZ2373+ZZ2373
+006662	754400		ZZ2373=ZZ2373+ZZ2373
+006662	731000		ZZ2373=ZZ2373+ZZ2373
+006662	662000		ZZ2373=ZZ2373+ZZ2373
+006662	544000		ZZ2373=ZZ2373+ZZ2373
+006662	013700		0 8192 -ZZ1373
+006663	544000		0 ZZ2373
-	 mark 2153, 106		/ 8 mono
+006664	000324		ZZ2374=ZZ2374+ZZ2374
+006664	000650		ZZ2374=ZZ2374+ZZ2374
+006664	001520		ZZ2374=ZZ2374+ZZ2374
+006664	003240		ZZ2374=ZZ2374+ZZ2374
+006664	006500		ZZ2374=ZZ2374+ZZ2374
+006664	015200		ZZ2374=ZZ2374+ZZ2374
+006664	032400		ZZ2374=ZZ2374+ZZ2374
+006664	065000		ZZ2374=ZZ2374+ZZ2374
+006664	013627		0 8192 -ZZ1374
+006665	065000		0 ZZ2374
-	 mark 2179, 462		/18 gemi
+006666	001634		ZZ2375=ZZ2375+ZZ2375
+006666	003470		ZZ2375=ZZ2375+ZZ2375
+006666	007160		ZZ2375=ZZ2375+ZZ2375
+006666	016340		ZZ2375=ZZ2375+ZZ2375
+006666	034700		ZZ2375=ZZ2375+ZZ2375
+006666	071600		ZZ2375=ZZ2375+ZZ2375
+006666	163400		ZZ2375=ZZ2375+ZZ2375
+006666	347000		ZZ2375=ZZ2375+ZZ2375
+006666	013575		0 8192 -ZZ1375
+006667	347000		0 ZZ2375
-	 mark 2179, -107	/10 mono
+006670	777450		ZZ2376=ZZ2376+ZZ2376
+006670	777120		ZZ2376=ZZ2376+ZZ2376
+006670	776240		ZZ2376=ZZ2376+ZZ2376
+006670	774500		ZZ2376=ZZ2376+ZZ2376
+006670	771200		ZZ2376=ZZ2376+ZZ2376
+006670	762400		ZZ2376=ZZ2376+ZZ2376
+006670	745000		ZZ2376=ZZ2376+ZZ2376
+006670	712000		ZZ2376=ZZ2376+ZZ2376
+006670	013575		0 8192 -ZZ1376
+006671	712000		0 ZZ2376
-	 mark 2184, -159	/11 mono
+006672	777300		ZZ2377=ZZ2377+ZZ2377
+006672	776600		ZZ2377=ZZ2377+ZZ2377
+006672	775400		ZZ2377=ZZ2377+ZZ2377
+006672	773000		ZZ2377=ZZ2377+ZZ2377
+006672	766000		ZZ2377=ZZ2377+ZZ2377
+006672	754000		ZZ2377=ZZ2377+ZZ2377
+006672	730000		ZZ2377=ZZ2377+ZZ2377
+006672	660000		ZZ2377=ZZ2377+ZZ2377
+006672	013570		0 8192 -ZZ1377
+006673	660000		0 ZZ2377
-	 mark 2204, 168		/13 mono
+006674	000520		ZZ2378=ZZ2378+ZZ2378
+006674	001240		ZZ2378=ZZ2378+ZZ2378
+006674	002500		ZZ2378=ZZ2378+ZZ2378
+006674	005200		ZZ2378=ZZ2378+ZZ2378
+006674	012400		ZZ2378=ZZ2378+ZZ2378
+006674	025000		ZZ2378=ZZ2378+ZZ2378
+006674	052000		ZZ2378=ZZ2378+ZZ2378
+006674	124000		ZZ2378=ZZ2378+ZZ2378
+006674	013544		0 8192 -ZZ1378
+006675	124000		0 ZZ2378
-	 mark 2232, -436	/ 7 cmaj
+006676	776226		ZZ2379=ZZ2379+ZZ2379
+006676	774454		ZZ2379=ZZ2379+ZZ2379
+006676	771130		ZZ2379=ZZ2379+ZZ2379
+006676	762260		ZZ2379=ZZ2379+ZZ2379
+006676	744540		ZZ2379=ZZ2379+ZZ2379
+006676	711300		ZZ2379=ZZ2379+ZZ2379
+006676	622600		ZZ2379=ZZ2379+ZZ2379
+006676	445400		ZZ2379=ZZ2379+ZZ2379
+006676	013510		0 8192 -ZZ1379
+006677	445400		0 ZZ2379
-	 mark 2239, -413	/ 8 cmaj
+006700	776304		ZZ2380=ZZ2380+ZZ2380
+006700	774610		ZZ2380=ZZ2380+ZZ2380
+006700	771420		ZZ2380=ZZ2380+ZZ2380
+006700	763040		ZZ2380=ZZ2380+ZZ2380
+006700	746100		ZZ2380=ZZ2380+ZZ2380
+006700	714200		ZZ2380=ZZ2380+ZZ2380
+006700	630400		ZZ2380=ZZ2380+ZZ2380
+006700	461000		ZZ2380=ZZ2380+ZZ2380
+006700	013501		0 8192 -ZZ1380
+006701	461000		0 ZZ2380
-	 mark 2245, -320	/
+006702	776576		ZZ2381=ZZ2381+ZZ2381
+006702	775374		ZZ2381=ZZ2381+ZZ2381
+006702	772770		ZZ2381=ZZ2381+ZZ2381
+006702	765760		ZZ2381=ZZ2381+ZZ2381
+006702	753740		ZZ2381=ZZ2381+ZZ2381
+006702	727700		ZZ2381=ZZ2381+ZZ2381
+006702	657600		ZZ2381=ZZ2381+ZZ2381
+006702	537400		ZZ2381=ZZ2381+ZZ2381
+006702	013473		0 8192 -ZZ1381
+006703	537400		0 ZZ2381
-	 mark 2250, 227		/15 mono
+006704	000706		ZZ2382=ZZ2382+ZZ2382
+006704	001614		ZZ2382=ZZ2382+ZZ2382
+006704	003430		ZZ2382=ZZ2382+ZZ2382
+006704	007060		ZZ2382=ZZ2382+ZZ2382
+006704	016140		ZZ2382=ZZ2382+ZZ2382
+006704	034300		ZZ2382=ZZ2382+ZZ2382
+006704	070600		ZZ2382=ZZ2382+ZZ2382
+006704	161400		ZZ2382=ZZ2382+ZZ2382
+006704	013466		0 8192 -ZZ1382
+006705	161400		0 ZZ2382
-	 mark 2266, 303		/30 gemi
+006706	001136		ZZ2383=ZZ2383+ZZ2383
+006706	002274		ZZ2383=ZZ2383+ZZ2383
+006706	004570		ZZ2383=ZZ2383+ZZ2383
+006706	011360		ZZ2383=ZZ2383+ZZ2383
+006706	022740		ZZ2383=ZZ2383+ZZ2383
+006706	045700		ZZ2383=ZZ2383+ZZ2383
+006706	113600		ZZ2383=ZZ2383+ZZ2383
+006706	227400		ZZ2383=ZZ2383+ZZ2383
+006706	013446		0 8192 -ZZ1383
+006707	227400		0 ZZ2383
-	 mark 2291, 57		/18 mono
+006710	000162		ZZ2384=ZZ2384+ZZ2384
+006710	000344		ZZ2384=ZZ2384+ZZ2384
+006710	000710		ZZ2384=ZZ2384+ZZ2384
+006710	001620		ZZ2384=ZZ2384+ZZ2384
+006710	003440		ZZ2384=ZZ2384+ZZ2384
+006710	007100		ZZ2384=ZZ2384+ZZ2384
+006710	016200		ZZ2384=ZZ2384+ZZ2384
+006710	034400		ZZ2384=ZZ2384+ZZ2384
+006710	013415		0 8192 -ZZ1384
+006711	034400		0 ZZ2384
-	 mark 2327, 303		/38 gemi
+006712	001136		ZZ2385=ZZ2385+ZZ2385
+006712	002274		ZZ2385=ZZ2385+ZZ2385
+006712	004570		ZZ2385=ZZ2385+ZZ2385
+006712	011360		ZZ2385=ZZ2385+ZZ2385
+006712	022740		ZZ2385=ZZ2385+ZZ2385
+006712	045700		ZZ2385=ZZ2385+ZZ2385
+006712	113600		ZZ2385=ZZ2385+ZZ2385
+006712	227400		ZZ2385=ZZ2385+ZZ2385
+006712	013351		0 8192 -ZZ1385
+006713	227400		0 ZZ2385
-	 mark 2328, -457	/15 cmaj
+006714	776154		ZZ2386=ZZ2386+ZZ2386
+006714	774330		ZZ2386=ZZ2386+ZZ2386
+006714	770660		ZZ2386=ZZ2386+ZZ2386
+006714	761540		ZZ2386=ZZ2386+ZZ2386
+006714	743300		ZZ2386=ZZ2386+ZZ2386
+006714	706600		ZZ2386=ZZ2386+ZZ2386
+006714	615400		ZZ2386=ZZ2386+ZZ2386
+006714	433000		ZZ2386=ZZ2386+ZZ2386
+006714	013350		0 8192 -ZZ1386
+006715	433000		0 ZZ2386
-	 mark 2330, -271	/14 cmaj
+006716	776740		ZZ2387=ZZ2387+ZZ2387
+006716	775700		ZZ2387=ZZ2387+ZZ2387
+006716	773600		ZZ2387=ZZ2387+ZZ2387
+006716	767400		ZZ2387=ZZ2387+ZZ2387
+006716	757000		ZZ2387=ZZ2387+ZZ2387
+006716	736000		ZZ2387=ZZ2387+ZZ2387
+006716	674000		ZZ2387=ZZ2387+ZZ2387
+006716	570000		ZZ2387=ZZ2387+ZZ2387
+006716	013346		0 8192 -ZZ1387
+006717	570000		0 ZZ2387
-	 mark 2340, -456	/19 cmaj
+006720	776156		ZZ2388=ZZ2388+ZZ2388
+006720	774334		ZZ2388=ZZ2388+ZZ2388
+006720	770670		ZZ2388=ZZ2388+ZZ2388
+006720	761560		ZZ2388=ZZ2388+ZZ2388
+006720	743340		ZZ2388=ZZ2388+ZZ2388
+006720	706700		ZZ2388=ZZ2388+ZZ2388
+006720	615600		ZZ2388=ZZ2388+ZZ2388
+006720	433400		ZZ2388=ZZ2388+ZZ2388
+006720	013334		0 8192 -ZZ1388
+006721	433400		0 ZZ2388
-	 mark 2342, -385	/20 cmaj
+006722	776374		ZZ2389=ZZ2389+ZZ2389
+006722	774770		ZZ2389=ZZ2389+ZZ2389
+006722	771760		ZZ2389=ZZ2389+ZZ2389
+006722	763740		ZZ2389=ZZ2389+ZZ2389
+006722	747700		ZZ2389=ZZ2389+ZZ2389
+006722	717600		ZZ2389=ZZ2389+ZZ2389
+006722	637400		ZZ2389=ZZ2389+ZZ2389
+006722	477000		ZZ2389=ZZ2389+ZZ2389
+006722	013332		0 8192 -ZZ1389
+006723	477000		0 ZZ2389
-	 mark 2378, -93		/19 mono
+006724	777504		ZZ2390=ZZ2390+ZZ2390
+006724	777210		ZZ2390=ZZ2390+ZZ2390
+006724	776420		ZZ2390=ZZ2390+ZZ2390
+006724	775040		ZZ2390=ZZ2390+ZZ2390
+006724	772100		ZZ2390=ZZ2390+ZZ2390
+006724	764200		ZZ2390=ZZ2390+ZZ2390
+006724	750400		ZZ2390=ZZ2390+ZZ2390
+006724	721000		ZZ2390=ZZ2390+ZZ2390
+006724	013266		0 8192 -ZZ1390
+006725	721000		0 ZZ2390
-	 mark 2379, 471		/43 gemi
+006726	001656		ZZ2391=ZZ2391+ZZ2391
+006726	003534		ZZ2391=ZZ2391+ZZ2391
+006726	007270		ZZ2391=ZZ2391+ZZ2391
+006726	016560		ZZ2391=ZZ2391+ZZ2391
+006726	035340		ZZ2391=ZZ2391+ZZ2391
+006726	072700		ZZ2391=ZZ2391+ZZ2391
+006726	165600		ZZ2391=ZZ2391+ZZ2391
+006726	353400		ZZ2391=ZZ2391+ZZ2391
+006726	013265		0 8192 -ZZ1391
+006727	353400		0 ZZ2391
-	 mark 2385, -352	/23 cmaj
+006730	776476		ZZ2392=ZZ2392+ZZ2392
+006730	775174		ZZ2392=ZZ2392+ZZ2392
+006730	772370		ZZ2392=ZZ2392+ZZ2392
+006730	764760		ZZ2392=ZZ2392+ZZ2392
+006730	751740		ZZ2392=ZZ2392+ZZ2392
+006730	723700		ZZ2392=ZZ2392+ZZ2392
+006730	647600		ZZ2392=ZZ2392+ZZ2392
+006730	517400		ZZ2392=ZZ2392+ZZ2392
+006730	013257		0 8192 -ZZ1392
+006731	517400		0 ZZ2392
-	 mark 2428, -8		/22 mono
+006732	777756		ZZ2393=ZZ2393+ZZ2393
+006732	777734		ZZ2393=ZZ2393+ZZ2393
+006732	777670		ZZ2393=ZZ2393+ZZ2393
+006732	777560		ZZ2393=ZZ2393+ZZ2393
+006732	777340		ZZ2393=ZZ2393+ZZ2393
+006732	776700		ZZ2393=ZZ2393+ZZ2393
+006732	775600		ZZ2393=ZZ2393+ZZ2393
+006732	773400		ZZ2393=ZZ2393+ZZ2393
+006732	013204		0 8192 -ZZ1393
+006733	773400		0 ZZ2393
-	 mark 2491, -429	/
+006734	776244		ZZ2394=ZZ2394+ZZ2394
+006734	774510		ZZ2394=ZZ2394+ZZ2394
+006734	771220		ZZ2394=ZZ2394+ZZ2394
+006734	762440		ZZ2394=ZZ2394+ZZ2394
+006734	745100		ZZ2394=ZZ2394+ZZ2394
+006734	712200		ZZ2394=ZZ2394+ZZ2394
+006734	624400		ZZ2394=ZZ2394+ZZ2394
+006734	451000		ZZ2394=ZZ2394+ZZ2394
+006734	013105		0 8192 -ZZ1394
+006735	451000		0 ZZ2394
-	 mark 2519, 208		/ 4 cmin
+006736	000640		ZZ2395=ZZ2395+ZZ2395
+006736	001500		ZZ2395=ZZ2395+ZZ2395
+006736	003200		ZZ2395=ZZ2395+ZZ2395
+006736	006400		ZZ2395=ZZ2395+ZZ2395
+006736	015000		ZZ2395=ZZ2395+ZZ2395
+006736	032000		ZZ2395=ZZ2395+ZZ2395
+006736	064000		ZZ2395=ZZ2395+ZZ2395
+006736	150000		ZZ2395=ZZ2395+ZZ2395
+006736	013051		0 8192 -ZZ1395
+006737	150000		0 ZZ2395
-	 mark 2527, 278		/ 6 cmin
+006740	001054		ZZ2396=ZZ2396+ZZ2396
+006740	002130		ZZ2396=ZZ2396+ZZ2396
+006740	004260		ZZ2396=ZZ2396+ZZ2396
+006740	010540		ZZ2396=ZZ2396+ZZ2396
+006740	021300		ZZ2396=ZZ2396+ZZ2396
+006740	042600		ZZ2396=ZZ2396+ZZ2396
+006740	105400		ZZ2396=ZZ2396+ZZ2396
+006740	213000		ZZ2396=ZZ2396+ZZ2396
+006740	013041		0 8192 -ZZ1396
+006741	213000		0 ZZ2396
-	 mark 2559, -503	/
+006742	776020		ZZ2397=ZZ2397+ZZ2397
+006742	774040		ZZ2397=ZZ2397+ZZ2397
+006742	770100		ZZ2397=ZZ2397+ZZ2397
+006742	760200		ZZ2397=ZZ2397+ZZ2397
+006742	740400		ZZ2397=ZZ2397+ZZ2397
+006742	701000		ZZ2397=ZZ2397+ZZ2397
+006742	602000		ZZ2397=ZZ2397+ZZ2397
+006742	404000		ZZ2397=ZZ2397+ZZ2397
+006742	013001		0 8192 -ZZ1397
+006743	404000		0 ZZ2397
-	 mark 2597, -212	/26 mono
+006744	777126		ZZ2398=ZZ2398+ZZ2398
+006744	776254		ZZ2398=ZZ2398+ZZ2398
+006744	774530		ZZ2398=ZZ2398+ZZ2398
+006744	771260		ZZ2398=ZZ2398+ZZ2398
+006744	762540		ZZ2398=ZZ2398+ZZ2398
+006744	745300		ZZ2398=ZZ2398+ZZ2398
+006744	712600		ZZ2398=ZZ2398+ZZ2398
+006744	625400		ZZ2398=ZZ2398+ZZ2398
+006744	012733		0 8192 -ZZ1398
+006745	625400		0 ZZ2398
-	 mark 2704, -412	/
+006746	776306		ZZ2399=ZZ2399+ZZ2399
+006746	774614		ZZ2399=ZZ2399+ZZ2399
+006746	771430		ZZ2399=ZZ2399+ZZ2399
+006746	763060		ZZ2399=ZZ2399+ZZ2399
+006746	746140		ZZ2399=ZZ2399+ZZ2399
+006746	714300		ZZ2399=ZZ2399+ZZ2399
+006746	630600		ZZ2399=ZZ2399+ZZ2399
+006746	461400		ZZ2399=ZZ2399+ZZ2399
+006746	012560		0 8192 -ZZ1399
+006747	461400		0 ZZ2399
-	 mark 2709, -25		/28 mono
+006750	777714		ZZ2400=ZZ2400+ZZ2400
+006750	777630		ZZ2400=ZZ2400+ZZ2400
+006750	777460		ZZ2400=ZZ2400+ZZ2400
+006750	777140		ZZ2400=ZZ2400+ZZ2400
+006750	776300		ZZ2400=ZZ2400+ZZ2400
+006750	774600		ZZ2400=ZZ2400+ZZ2400
+006750	771400		ZZ2400=ZZ2400+ZZ2400
+006750	763000		ZZ2400=ZZ2400+ZZ2400
+006750	012553		0 8192 -ZZ1400
+006751	763000		0 ZZ2400
-	 mark 2714, 60		/
+006752	000170		ZZ2401=ZZ2401+ZZ2401
+006752	000360		ZZ2401=ZZ2401+ZZ2401
+006752	000740		ZZ2401=ZZ2401+ZZ2401
+006752	001700		ZZ2401=ZZ2401+ZZ2401
+006752	003600		ZZ2401=ZZ2401+ZZ2401
+006752	007400		ZZ2401=ZZ2401+ZZ2401
+006752	017000		ZZ2401=ZZ2401+ZZ2401
+006752	036000		ZZ2401=ZZ2401+ZZ2401
+006752	012546		0 8192 -ZZ1401
+006753	036000		0 ZZ2401
-	 mark 2751, -61		/29 mono
+006754	777604		ZZ2402=ZZ2402+ZZ2402
+006754	777410		ZZ2402=ZZ2402+ZZ2402
+006754	777020		ZZ2402=ZZ2402+ZZ2402
+006754	776040		ZZ2402=ZZ2402+ZZ2402
+006754	774100		ZZ2402=ZZ2402+ZZ2402
+006754	770200		ZZ2402=ZZ2402+ZZ2402
+006754	760400		ZZ2402=ZZ2402+ZZ2402
+006754	741000		ZZ2402=ZZ2402+ZZ2402
+006754	012501		0 8192 -ZZ1402
+006755	741000		0 ZZ2402
-	 mark 2757, -431	/16 pupp
+006756	776240		ZZ2403=ZZ2403+ZZ2403
+006756	774500		ZZ2403=ZZ2403+ZZ2403
+006756	771200		ZZ2403=ZZ2403+ZZ2403
+006756	762400		ZZ2403=ZZ2403+ZZ2403
+006756	745000		ZZ2403=ZZ2403+ZZ2403
+006756	712000		ZZ2403=ZZ2403+ZZ2403
+006756	624000		ZZ2403=ZZ2403+ZZ2403
+006756	450000		ZZ2403=ZZ2403+ZZ2403
+006756	012473		0 8192 -ZZ1403
+006757	450000		0 ZZ2403
-	 mark 2768, -288	/19 pupp
+006760	776676		ZZ2404=ZZ2404+ZZ2404
+006760	775574		ZZ2404=ZZ2404+ZZ2404
+006760	773370		ZZ2404=ZZ2404+ZZ2404
+006760	766760		ZZ2404=ZZ2404+ZZ2404
+006760	755740		ZZ2404=ZZ2404+ZZ2404
+006760	733700		ZZ2404=ZZ2404+ZZ2404
+006760	667600		ZZ2404=ZZ2404+ZZ2404
+006760	557400		ZZ2404=ZZ2404+ZZ2404
+006760	012460		0 8192 -ZZ1404
+006761	557400		0 ZZ2404
-	 mark 2794, 216		/17 canc
+006762	000660		ZZ2405=ZZ2405+ZZ2405
+006762	001540		ZZ2405=ZZ2405+ZZ2405
+006762	003300		ZZ2405=ZZ2405+ZZ2405
+006762	006600		ZZ2405=ZZ2405+ZZ2405
+006762	015400		ZZ2405=ZZ2405+ZZ2405
+006762	033000		ZZ2405=ZZ2405+ZZ2405
+006762	066000		ZZ2405=ZZ2405+ZZ2405
+006762	154000		ZZ2405=ZZ2405+ZZ2405
+006762	012426		0 8192 -ZZ1405
+006763	154000		0 ZZ2405
-	 mark 2848, -82		/
+006764	777532		ZZ2406=ZZ2406+ZZ2406
+006764	777264		ZZ2406=ZZ2406+ZZ2406
+006764	776550		ZZ2406=ZZ2406+ZZ2406
+006764	775320		ZZ2406=ZZ2406+ZZ2406
+006764	772640		ZZ2406=ZZ2406+ZZ2406
+006764	765500		ZZ2406=ZZ2406+ZZ2406
+006764	753200		ZZ2406=ZZ2406+ZZ2406
+006764	726400		ZZ2406=ZZ2406+ZZ2406
+006764	012340		0 8192 -ZZ1406
+006765	726400		0 ZZ2406
-	 mark 2915, 138		/ 4 hyda
+006766	000424		ZZ2407=ZZ2407+ZZ2407
+006766	001050		ZZ2407=ZZ2407+ZZ2407
+006766	002120		ZZ2407=ZZ2407+ZZ2407
+006766	004240		ZZ2407=ZZ2407+ZZ2407
+006766	010500		ZZ2407=ZZ2407+ZZ2407
+006766	021200		ZZ2407=ZZ2407+ZZ2407
+006766	042400		ZZ2407=ZZ2407+ZZ2407
+006766	105000		ZZ2407=ZZ2407+ZZ2407
+006766	012235		0 8192 -ZZ1407
+006767	105000		0 ZZ2407
-	 mark 2921, 84		/ 5 hyda
+006770	000250		ZZ2408=ZZ2408+ZZ2408
+006770	000520		ZZ2408=ZZ2408+ZZ2408
+006770	001240		ZZ2408=ZZ2408+ZZ2408
+006770	002500		ZZ2408=ZZ2408+ZZ2408
+006770	005200		ZZ2408=ZZ2408+ZZ2408
+006770	012400		ZZ2408=ZZ2408+ZZ2408
+006770	025000		ZZ2408=ZZ2408+ZZ2408
+006770	052000		ZZ2408=ZZ2408+ZZ2408
+006770	012227		0 8192 -ZZ1408
+006771	052000		0 ZZ2408
-	 mark 2942, -355	/ 9 hyda
+006772	776470		ZZ2409=ZZ2409+ZZ2409
+006772	775160		ZZ2409=ZZ2409+ZZ2409
+006772	772340		ZZ2409=ZZ2409+ZZ2409
+006772	764700		ZZ2409=ZZ2409+ZZ2409
+006772	751600		ZZ2409=ZZ2409+ZZ2409
+006772	723400		ZZ2409=ZZ2409+ZZ2409
+006772	647000		ZZ2409=ZZ2409+ZZ2409
+006772	516000		ZZ2409=ZZ2409+ZZ2409
+006772	012202		0 8192 -ZZ1409
+006773	516000		0 ZZ2409
-	 mark 2944, 497		/43 canc
+006774	001742		ZZ2410=ZZ2410+ZZ2410
+006774	003704		ZZ2410=ZZ2410+ZZ2410
+006774	007610		ZZ2410=ZZ2410+ZZ2410
+006774	017420		ZZ2410=ZZ2410+ZZ2410
+006774	037040		ZZ2410=ZZ2410+ZZ2410
+006774	076100		ZZ2410=ZZ2410+ZZ2410
+006774	174200		ZZ2410=ZZ2410+ZZ2410
+006774	370400		ZZ2410=ZZ2410+ZZ2410
+006774	012200		0 8192 -ZZ1410
+006775	370400		0 ZZ2410
-	 mark 2947, 85		/ 7 hyda
+006776	000252		ZZ2411=ZZ2411+ZZ2411
+006776	000524		ZZ2411=ZZ2411+ZZ2411
+006776	001250		ZZ2411=ZZ2411+ZZ2411
+006776	002520		ZZ2411=ZZ2411+ZZ2411
+006776	005240		ZZ2411=ZZ2411+ZZ2411
+006776	012500		ZZ2411=ZZ2411+ZZ2411
+006776	025200		ZZ2411=ZZ2411+ZZ2411
+006776	052400		ZZ2411=ZZ2411+ZZ2411
+006776	012175		0 8192 -ZZ1411
+006777	052400		0 ZZ2411
-	 mark 2951, -156	/
+007000	777306		ZZ2412=ZZ2412+ZZ2412
+007000	776614		ZZ2412=ZZ2412+ZZ2412
+007000	775430		ZZ2412=ZZ2412+ZZ2412
+007000	773060		ZZ2412=ZZ2412+ZZ2412
+007000	766140		ZZ2412=ZZ2412+ZZ2412
+007000	754300		ZZ2412=ZZ2412+ZZ2412
+007000	730600		ZZ2412=ZZ2412+ZZ2412
+007000	661400		ZZ2412=ZZ2412+ZZ2412
+007000	012171		0 8192 -ZZ1412
+007001	661400		0 ZZ2412
-	 mark 2953, 421		/47 canc
+007002	001512		ZZ2413=ZZ2413+ZZ2413
+007002	003224		ZZ2413=ZZ2413+ZZ2413
+007002	006450		ZZ2413=ZZ2413+ZZ2413
+007002	015120		ZZ2413=ZZ2413+ZZ2413
+007002	032240		ZZ2413=ZZ2413+ZZ2413
+007002	064500		ZZ2413=ZZ2413+ZZ2413
+007002	151200		ZZ2413=ZZ2413+ZZ2413
+007002	322400		ZZ2413=ZZ2413+ZZ2413
+007002	012167		0 8192 -ZZ1413
+007003	322400		0 ZZ2413
-	 mark 2968, -300	/12 hyda
+007004	776646		ZZ2414=ZZ2414+ZZ2414
+007004	775514		ZZ2414=ZZ2414+ZZ2414
+007004	773230		ZZ2414=ZZ2414+ZZ2414
+007004	766460		ZZ2414=ZZ2414+ZZ2414
+007004	755140		ZZ2414=ZZ2414+ZZ2414
+007004	732300		ZZ2414=ZZ2414+ZZ2414
+007004	664600		ZZ2414=ZZ2414+ZZ2414
+007004	551400		ZZ2414=ZZ2414+ZZ2414
+007004	012150		0 8192 -ZZ1414
+007005	551400		0 ZZ2414
-	 mark 2976, 141		/13 hyda
+007006	000432		ZZ2415=ZZ2415+ZZ2415
+007006	001064		ZZ2415=ZZ2415+ZZ2415
+007006	002150		ZZ2415=ZZ2415+ZZ2415
+007006	004320		ZZ2415=ZZ2415+ZZ2415
+007006	010640		ZZ2415=ZZ2415+ZZ2415
+007006	021500		ZZ2415=ZZ2415+ZZ2415
+007006	043200		ZZ2415=ZZ2415+ZZ2415
+007006	106400		ZZ2415=ZZ2415+ZZ2415
+007006	012140		0 8192 -ZZ1415
+007007	106400		0 ZZ2415
-	 mark 3032, 279		/65 canc
+007010	001056		ZZ2416=ZZ2416+ZZ2416
+007010	002134		ZZ2416=ZZ2416+ZZ2416
+007010	004270		ZZ2416=ZZ2416+ZZ2416
+007010	010560		ZZ2416=ZZ2416+ZZ2416
+007010	021340		ZZ2416=ZZ2416+ZZ2416
+007010	042700		ZZ2416=ZZ2416+ZZ2416
+007010	105600		ZZ2416=ZZ2416+ZZ2416
+007010	213400		ZZ2416=ZZ2416+ZZ2416
+007010	012050		0 8192 -ZZ1416
+007011	213400		0 ZZ2416
-	 mark 3124, 62		/22 hyda
+007012	000174		ZZ2417=ZZ2417+ZZ2417
+007012	000370		ZZ2417=ZZ2417+ZZ2417
+007012	000760		ZZ2417=ZZ2417+ZZ2417
+007012	001740		ZZ2417=ZZ2417+ZZ2417
+007012	003700		ZZ2417=ZZ2417+ZZ2417
+007012	007600		ZZ2417=ZZ2417+ZZ2417
+007012	017400		ZZ2417=ZZ2417+ZZ2417
+007012	037000		ZZ2417=ZZ2417+ZZ2417
+007012	011714		0 8192 -ZZ1417
+007013	037000		0 ZZ2417
-	 mark 3157, -263	/26 hyda
+007014	776760		ZZ2418=ZZ2418+ZZ2418
+007014	775740		ZZ2418=ZZ2418+ZZ2418
+007014	773700		ZZ2418=ZZ2418+ZZ2418
+007014	767600		ZZ2418=ZZ2418+ZZ2418
+007014	757400		ZZ2418=ZZ2418+ZZ2418
+007014	737000		ZZ2418=ZZ2418+ZZ2418
+007014	676000		ZZ2418=ZZ2418+ZZ2418
+007014	574000		ZZ2418=ZZ2418+ZZ2418
+007014	011653		0 8192 -ZZ1418
+007015	574000		0 ZZ2418
-	 mark 3161, -208	/27 hyda
+007016	777136		ZZ2419=ZZ2419+ZZ2419
+007016	776274		ZZ2419=ZZ2419+ZZ2419
+007016	774570		ZZ2419=ZZ2419+ZZ2419
+007016	771360		ZZ2419=ZZ2419+ZZ2419
+007016	762740		ZZ2419=ZZ2419+ZZ2419
+007016	745700		ZZ2419=ZZ2419+ZZ2419
+007016	713600		ZZ2419=ZZ2419+ZZ2419
+007016	627400		ZZ2419=ZZ2419+ZZ2419
+007016	011647		0 8192 -ZZ1419
+007017	627400		0 ZZ2419
-	 mark 3209, -53		/31 hyda
+007020	777624		ZZ2420=ZZ2420+ZZ2420
+007020	777450		ZZ2420=ZZ2420+ZZ2420
+007020	777120		ZZ2420=ZZ2420+ZZ2420
+007020	776240		ZZ2420=ZZ2420+ZZ2420
+007020	774500		ZZ2420=ZZ2420+ZZ2420
+007020	771200		ZZ2420=ZZ2420+ZZ2420
+007020	762400		ZZ2420=ZZ2420+ZZ2420
+007020	745000		ZZ2420=ZZ2420+ZZ2420
+007020	011567		0 8192 -ZZ1420
+007021	745000		0 ZZ2420
-	 mark 3225, -17		/32 hyda
+007022	777734		ZZ2421=ZZ2421+ZZ2421
+007022	777670		ZZ2421=ZZ2421+ZZ2421
+007022	777560		ZZ2421=ZZ2421+ZZ2421
+007022	777340		ZZ2421=ZZ2421+ZZ2421
+007022	776700		ZZ2421=ZZ2421+ZZ2421
+007022	775600		ZZ2421=ZZ2421+ZZ2421
+007022	773400		ZZ2421=ZZ2421+ZZ2421
+007022	767000		ZZ2421=ZZ2421+ZZ2421
+007022	011547		0 8192 -ZZ1421
+007023	767000		0 ZZ2421
-	 mark 3261, 116		/
+007024	000350		ZZ2422=ZZ2422+ZZ2422
+007024	000720		ZZ2422=ZZ2422+ZZ2422
+007024	001640		ZZ2422=ZZ2422+ZZ2422
+007024	003500		ZZ2422=ZZ2422+ZZ2422
+007024	007200		ZZ2422=ZZ2422+ZZ2422
+007024	016400		ZZ2422=ZZ2422+ZZ2422
+007024	035000		ZZ2422=ZZ2422+ZZ2422
+007024	072000		ZZ2422=ZZ2422+ZZ2422
+007024	011503		0 8192 -ZZ1422
+007025	072000		0 ZZ2422
-	 mark 3270, -16		/35 hyda
+007026	777736		ZZ2423=ZZ2423+ZZ2423
+007026	777674		ZZ2423=ZZ2423+ZZ2423
+007026	777570		ZZ2423=ZZ2423+ZZ2423
+007026	777360		ZZ2423=ZZ2423+ZZ2423
+007026	776740		ZZ2423=ZZ2423+ZZ2423
+007026	775700		ZZ2423=ZZ2423+ZZ2423
+007026	773600		ZZ2423=ZZ2423+ZZ2423
+007026	767400		ZZ2423=ZZ2423+ZZ2423
+007026	011472		0 8192 -ZZ1423
+007027	767400		0 ZZ2423
-	 mark 3274, -316	/38 hyda
+007030	776606		ZZ2424=ZZ2424+ZZ2424
+007030	775414		ZZ2424=ZZ2424+ZZ2424
+007030	773030		ZZ2424=ZZ2424+ZZ2424
+007030	766060		ZZ2424=ZZ2424+ZZ2424
+007030	754140		ZZ2424=ZZ2424+ZZ2424
+007030	730300		ZZ2424=ZZ2424+ZZ2424
+007030	660600		ZZ2424=ZZ2424+ZZ2424
+007030	541400		ZZ2424=ZZ2424+ZZ2424
+007030	011466		0 8192 -ZZ1424
+007031	541400		0 ZZ2424
-	 mark 3276, 236		/14 leon
+007032	000730		ZZ2425=ZZ2425+ZZ2425
+007032	001660		ZZ2425=ZZ2425+ZZ2425
+007032	003540		ZZ2425=ZZ2425+ZZ2425
+007032	007300		ZZ2425=ZZ2425+ZZ2425
+007032	016600		ZZ2425=ZZ2425+ZZ2425
+007032	035400		ZZ2425=ZZ2425+ZZ2425
+007032	073000		ZZ2425=ZZ2425+ZZ2425
+007032	166000		ZZ2425=ZZ2425+ZZ2425
+007032	011464		0 8192 -ZZ1425
+007033	166000		0 ZZ2425
-	 mark 3338, -327	/39 hyda
+007034	776560		ZZ2426=ZZ2426+ZZ2426
+007034	775340		ZZ2426=ZZ2426+ZZ2426
+007034	772700		ZZ2426=ZZ2426+ZZ2426
+007034	765600		ZZ2426=ZZ2426+ZZ2426
+007034	753400		ZZ2426=ZZ2426+ZZ2426
+007034	727000		ZZ2426=ZZ2426+ZZ2426
+007034	656000		ZZ2426=ZZ2426+ZZ2426
+007034	534000		ZZ2426=ZZ2426+ZZ2426
+007034	011366		0 8192 -ZZ1426
+007035	534000		0 ZZ2426
-	 mark 3385, 194		/29 leon
+007036	000604		ZZ2427=ZZ2427+ZZ2427
+007036	001410		ZZ2427=ZZ2427+ZZ2427
+007036	003020		ZZ2427=ZZ2427+ZZ2427
+007036	006040		ZZ2427=ZZ2427+ZZ2427
+007036	014100		ZZ2427=ZZ2427+ZZ2427
+007036	030200		ZZ2427=ZZ2427+ZZ2427
+007036	060400		ZZ2427=ZZ2427+ZZ2427
+007036	141000		ZZ2427=ZZ2427+ZZ2427
+007036	011307		0 8192 -ZZ1427
+007037	141000		0 ZZ2427
-	 mark 3415, -286	/40 hyda
+007040	776702		ZZ2428=ZZ2428+ZZ2428
+007040	775604		ZZ2428=ZZ2428+ZZ2428
+007040	773410		ZZ2428=ZZ2428+ZZ2428
+007040	767020		ZZ2428=ZZ2428+ZZ2428
+007040	756040		ZZ2428=ZZ2428+ZZ2428
+007040	734100		ZZ2428=ZZ2428+ZZ2428
+007040	670200		ZZ2428=ZZ2428+ZZ2428
+007040	560400		ZZ2428=ZZ2428+ZZ2428
+007040	011251		0 8192 -ZZ1428
+007041	560400		0 ZZ2428
-	 mark 3428, 239		/31 leon
+007042	000736		ZZ2429=ZZ2429+ZZ2429
+007042	001674		ZZ2429=ZZ2429+ZZ2429
+007042	003570		ZZ2429=ZZ2429+ZZ2429
+007042	007360		ZZ2429=ZZ2429+ZZ2429
+007042	016740		ZZ2429=ZZ2429+ZZ2429
+007042	035700		ZZ2429=ZZ2429+ZZ2429
+007042	073600		ZZ2429=ZZ2429+ZZ2429
+007042	167400		ZZ2429=ZZ2429+ZZ2429
+007042	011234		0 8192 -ZZ1429
+007043	167400		0 ZZ2429
-	 mark 3429, 3		/15 sext
+007044	000006		ZZ2430=ZZ2430+ZZ2430
+007044	000014		ZZ2430=ZZ2430+ZZ2430
+007044	000030		ZZ2430=ZZ2430+ZZ2430
+007044	000060		ZZ2430=ZZ2430+ZZ2430
+007044	000140		ZZ2430=ZZ2430+ZZ2430
+007044	000300		ZZ2430=ZZ2430+ZZ2430
+007044	000600		ZZ2430=ZZ2430+ZZ2430
+007044	001400		ZZ2430=ZZ2430+ZZ2430
+007044	011233		0 8192 -ZZ1430
+007045	001400		0 ZZ2430
-	 mark 3446, -270	/41 hyda
+007046	776742		ZZ2431=ZZ2431+ZZ2431
+007046	775704		ZZ2431=ZZ2431+ZZ2431
+007046	773610		ZZ2431=ZZ2431+ZZ2431
+007046	767420		ZZ2431=ZZ2431+ZZ2431
+007046	757040		ZZ2431=ZZ2431+ZZ2431
+007046	736100		ZZ2431=ZZ2431+ZZ2431
+007046	674200		ZZ2431=ZZ2431+ZZ2431
+007046	570400		ZZ2431=ZZ2431+ZZ2431
+007046	011212		0 8192 -ZZ1431
+007047	570400		0 ZZ2431
-	 mark 3495, 455		/40 leon
+007050	001616		ZZ2432=ZZ2432+ZZ2432
+007050	003434		ZZ2432=ZZ2432+ZZ2432
+007050	007070		ZZ2432=ZZ2432+ZZ2432
+007050	016160		ZZ2432=ZZ2432+ZZ2432
+007050	034340		ZZ2432=ZZ2432+ZZ2432
+007050	070700		ZZ2432=ZZ2432+ZZ2432
+007050	161600		ZZ2432=ZZ2432+ZZ2432
+007050	343400		ZZ2432=ZZ2432+ZZ2432
+007050	011131		0 8192 -ZZ1432
+007051	343400		0 ZZ2432
-	 mark 3534, -372	/42 hyda
+007052	776426		ZZ2433=ZZ2433+ZZ2433
+007052	775054		ZZ2433=ZZ2433+ZZ2433
+007052	772130		ZZ2433=ZZ2433+ZZ2433
+007052	764260		ZZ2433=ZZ2433+ZZ2433
+007052	750540		ZZ2433=ZZ2433+ZZ2433
+007052	721300		ZZ2433=ZZ2433+ZZ2433
+007052	642600		ZZ2433=ZZ2433+ZZ2433
+007052	505400		ZZ2433=ZZ2433+ZZ2433
+007052	011062		0 8192 -ZZ1433
+007053	505400		0 ZZ2433
-	 mark 3557, -3		/30 sext
+007054	777770		ZZ2434=ZZ2434+ZZ2434
+007054	777760		ZZ2434=ZZ2434+ZZ2434
+007054	777740		ZZ2434=ZZ2434+ZZ2434
+007054	777700		ZZ2434=ZZ2434+ZZ2434
+007054	777600		ZZ2434=ZZ2434+ZZ2434
+007054	777400		ZZ2434=ZZ2434+ZZ2434
+007054	777000		ZZ2434=ZZ2434+ZZ2434
+007054	776000		ZZ2434=ZZ2434+ZZ2434
+007054	011033		0 8192 -ZZ1434
+007055	776000		0 ZZ2434
-	 mark 3570, 223		/47 leon
+007056	000676		ZZ2435=ZZ2435+ZZ2435
+007056	001574		ZZ2435=ZZ2435+ZZ2435
+007056	003370		ZZ2435=ZZ2435+ZZ2435
+007056	006760		ZZ2435=ZZ2435+ZZ2435
+007056	015740		ZZ2435=ZZ2435+ZZ2435
+007056	033700		ZZ2435=ZZ2435+ZZ2435
+007056	067600		ZZ2435=ZZ2435+ZZ2435
+007056	157400		ZZ2435=ZZ2435+ZZ2435
+007056	011016		0 8192 -ZZ1435
+007057	157400		0 ZZ2435
-	 mark 3726, -404	/al crat
+007060	776326		ZZ2436=ZZ2436+ZZ2436
+007060	774654		ZZ2436=ZZ2436+ZZ2436
+007060	771530		ZZ2436=ZZ2436+ZZ2436
+007060	763260		ZZ2436=ZZ2436+ZZ2436
+007060	746540		ZZ2436=ZZ2436+ZZ2436
+007060	715300		ZZ2436=ZZ2436+ZZ2436
+007060	632600		ZZ2436=ZZ2436+ZZ2436
+007060	465400		ZZ2436=ZZ2436+ZZ2436
+007060	010562		0 8192 -ZZ1436
+007061	465400		0 ZZ2436
-	 mark 3736, -44		/61 leon
+007062	777646		ZZ2437=ZZ2437+ZZ2437
+007062	777514		ZZ2437=ZZ2437+ZZ2437
+007062	777230		ZZ2437=ZZ2437+ZZ2437
+007062	776460		ZZ2437=ZZ2437+ZZ2437
+007062	775140		ZZ2437=ZZ2437+ZZ2437
+007062	772300		ZZ2437=ZZ2437+ZZ2437
+007062	764600		ZZ2437=ZZ2437+ZZ2437
+007062	751400		ZZ2437=ZZ2437+ZZ2437
+007062	010550		0 8192 -ZZ1437
+007063	751400		0 ZZ2437
-	 mark 3738, 471		/60 leon
+007064	001656		ZZ2438=ZZ2438+ZZ2438
+007064	003534		ZZ2438=ZZ2438+ZZ2438
+007064	007270		ZZ2438=ZZ2438+ZZ2438
+007064	016560		ZZ2438=ZZ2438+ZZ2438
+007064	035340		ZZ2438=ZZ2438+ZZ2438
+007064	072700		ZZ2438=ZZ2438+ZZ2438
+007064	165600		ZZ2438=ZZ2438+ZZ2438
+007064	353400		ZZ2438=ZZ2438+ZZ2438
+007064	010546		0 8192 -ZZ1438
+007065	353400		0 ZZ2438
-	 mark 3754, 179		/63 leon
+007066	000546		ZZ2439=ZZ2439+ZZ2439
+007066	001314		ZZ2439=ZZ2439+ZZ2439
+007066	002630		ZZ2439=ZZ2439+ZZ2439
+007066	005460		ZZ2439=ZZ2439+ZZ2439
+007066	013140		ZZ2439=ZZ2439+ZZ2439
+007066	026300		ZZ2439=ZZ2439+ZZ2439
+007066	054600		ZZ2439=ZZ2439+ZZ2439
+007066	131400		ZZ2439=ZZ2439+ZZ2439
+007066	010526		0 8192 -ZZ1439
+007067	131400		0 ZZ2439
-	 mark 3793, -507	/11 crat
+007070	776010		ZZ2440=ZZ2440+ZZ2440
+007070	774020		ZZ2440=ZZ2440+ZZ2440
+007070	770040		ZZ2440=ZZ2440+ZZ2440
+007070	760100		ZZ2440=ZZ2440+ZZ2440
+007070	740200		ZZ2440=ZZ2440+ZZ2440
+007070	700400		ZZ2440=ZZ2440+ZZ2440
+007070	601000		ZZ2440=ZZ2440+ZZ2440
+007070	402000		ZZ2440=ZZ2440+ZZ2440
+007070	010457		0 8192 -ZZ1440
+007071	402000		0 ZZ2440
-	 mark 3821, -71		/74 leon
+007072	777560		ZZ2441=ZZ2441+ZZ2441
+007072	777340		ZZ2441=ZZ2441+ZZ2441
+007072	776700		ZZ2441=ZZ2441+ZZ2441
+007072	775600		ZZ2441=ZZ2441+ZZ2441
+007072	773400		ZZ2441=ZZ2441+ZZ2441
+007072	767000		ZZ2441=ZZ2441+ZZ2441
+007072	756000		ZZ2441=ZZ2441+ZZ2441
+007072	734000		ZZ2441=ZZ2441+ZZ2441
+007072	010423		0 8192 -ZZ1441
+007073	734000		0 ZZ2441
-	 mark 3836, -324	/12 crat
+007074	776566		ZZ2442=ZZ2442+ZZ2442
+007074	775354		ZZ2442=ZZ2442+ZZ2442
+007074	772730		ZZ2442=ZZ2442+ZZ2442
+007074	765660		ZZ2442=ZZ2442+ZZ2442
+007074	753540		ZZ2442=ZZ2442+ZZ2442
+007074	727300		ZZ2442=ZZ2442+ZZ2442
+007074	656600		ZZ2442=ZZ2442+ZZ2442
+007074	535400		ZZ2442=ZZ2442+ZZ2442
+007074	010404		0 8192 -ZZ1442
+007075	535400		0 ZZ2442
-	 mark 3846, 150		/77 leon
+007076	000454		ZZ2443=ZZ2443+ZZ2443
+007076	001130		ZZ2443=ZZ2443+ZZ2443
+007076	002260		ZZ2443=ZZ2443+ZZ2443
+007076	004540		ZZ2443=ZZ2443+ZZ2443
+007076	011300		ZZ2443=ZZ2443+ZZ2443
+007076	022600		ZZ2443=ZZ2443+ZZ2443
+007076	045400		ZZ2443=ZZ2443+ZZ2443
+007076	113000		ZZ2443=ZZ2443+ZZ2443
+007076	010372		0 8192 -ZZ1443
+007077	113000		0 ZZ2443
-	 mark 3861, 252		/78 leon
+007100	000770		ZZ2444=ZZ2444+ZZ2444
+007100	001760		ZZ2444=ZZ2444+ZZ2444
+007100	003740		ZZ2444=ZZ2444+ZZ2444
+007100	007700		ZZ2444=ZZ2444+ZZ2444
+007100	017600		ZZ2444=ZZ2444+ZZ2444
+007100	037400		ZZ2444=ZZ2444+ZZ2444
+007100	077000		ZZ2444=ZZ2444+ZZ2444
+007100	176000		ZZ2444=ZZ2444+ZZ2444
+007100	010353		0 8192 -ZZ1444
+007101	176000		0 ZZ2444
-	 mark 3868, -390	/15 crat
+007102	776362		ZZ2445=ZZ2445+ZZ2445
+007102	774744		ZZ2445=ZZ2445+ZZ2445
+007102	771710		ZZ2445=ZZ2445+ZZ2445
+007102	763620		ZZ2445=ZZ2445+ZZ2445
+007102	747440		ZZ2445=ZZ2445+ZZ2445
+007102	717100		ZZ2445=ZZ2445+ZZ2445
+007102	636200		ZZ2445=ZZ2445+ZZ2445
+007102	474400		ZZ2445=ZZ2445+ZZ2445
+007102	010344		0 8192 -ZZ1445
+007103	474400		0 ZZ2445
-	 mark 3935, -211	/21 crat
+007104	777130		ZZ2446=ZZ2446+ZZ2446
+007104	776260		ZZ2446=ZZ2446+ZZ2446
+007104	774540		ZZ2446=ZZ2446+ZZ2446
+007104	771300		ZZ2446=ZZ2446+ZZ2446
+007104	762600		ZZ2446=ZZ2446+ZZ2446
+007104	745400		ZZ2446=ZZ2446+ZZ2446
+007104	713000		ZZ2446=ZZ2446+ZZ2446
+007104	626000		ZZ2446=ZZ2446+ZZ2446
+007104	010241		0 8192 -ZZ1446
+007105	626000		0 ZZ2446
-	 mark 3936, -6 		/91 leon
+007106	777762		ZZ2447=ZZ2447+ZZ2447
+007106	777744		ZZ2447=ZZ2447+ZZ2447
+007106	777710		ZZ2447=ZZ2447+ZZ2447
+007106	777620		ZZ2447=ZZ2447+ZZ2447
+007106	777440		ZZ2447=ZZ2447+ZZ2447
+007106	777100		ZZ2447=ZZ2447+ZZ2447
+007106	776200		ZZ2447=ZZ2447+ZZ2447
+007106	774400		ZZ2447=ZZ2447+ZZ2447
+007106	010240		0 8192 -ZZ1447
+007107	774400		0 ZZ2447
-	 mark 3981, -405	/27 crat
+007110	776324		ZZ2448=ZZ2448+ZZ2448
+007110	774650		ZZ2448=ZZ2448+ZZ2448
+007110	771520		ZZ2448=ZZ2448+ZZ2448
+007110	763240		ZZ2448=ZZ2448+ZZ2448
+007110	746500		ZZ2448=ZZ2448+ZZ2448
+007110	715200		ZZ2448=ZZ2448+ZZ2448
+007110	632400		ZZ2448=ZZ2448+ZZ2448
+007110	465000		ZZ2448=ZZ2448+ZZ2448
+007110	010163		0 8192 -ZZ1448
+007111	465000		0 ZZ2448
-	 mark 3986, 161		/ 3 virg
+007112	000502		ZZ2449=ZZ2449+ZZ2449
+007112	001204		ZZ2449=ZZ2449+ZZ2449
+007112	002410		ZZ2449=ZZ2449+ZZ2449
+007112	005020		ZZ2449=ZZ2449+ZZ2449
+007112	012040		ZZ2449=ZZ2449+ZZ2449
+007112	024100		ZZ2449=ZZ2449+ZZ2449
+007112	050200		ZZ2449=ZZ2449+ZZ2449
+007112	120400		ZZ2449=ZZ2449+ZZ2449
+007112	010156		0 8192 -ZZ1449
+007113	120400		0 ZZ2449
-	 mark 3998, 473		/93 leon
+007114	001662		ZZ2450=ZZ2450+ZZ2450
+007114	003544		ZZ2450=ZZ2450+ZZ2450
+007114	007310		ZZ2450=ZZ2450+ZZ2450
+007114	016620		ZZ2450=ZZ2450+ZZ2450
+007114	035440		ZZ2450=ZZ2450+ZZ2450
+007114	073100		ZZ2450=ZZ2450+ZZ2450
+007114	166200		ZZ2450=ZZ2450+ZZ2450
+007114	354400		ZZ2450=ZZ2450+ZZ2450
+007114	010142		0 8192 -ZZ1450
+007115	354400		0 ZZ2450
-	 mark 4013, 53		/ 5 virg
+007116	000152		ZZ2451=ZZ2451+ZZ2451
+007116	000324		ZZ2451=ZZ2451+ZZ2451
+007116	000650		ZZ2451=ZZ2451+ZZ2451
+007116	001520		ZZ2451=ZZ2451+ZZ2451
+007116	003240		ZZ2451=ZZ2451+ZZ2451
+007116	006500		ZZ2451=ZZ2451+ZZ2451
+007116	015200		ZZ2451=ZZ2451+ZZ2451
+007116	032400		ZZ2451=ZZ2451+ZZ2451
+007116	010123		0 8192 -ZZ1451
+007117	032400		0 ZZ2451
-	 mark 4072, 163		/ 8 virg
+007120	000506		ZZ2452=ZZ2452+ZZ2452
+007120	001214		ZZ2452=ZZ2452+ZZ2452
+007120	002430		ZZ2452=ZZ2452+ZZ2452
+007120	005060		ZZ2452=ZZ2452+ZZ2452
+007120	012140		ZZ2452=ZZ2452+ZZ2452
+007120	024300		ZZ2452=ZZ2452+ZZ2452
+007120	050600		ZZ2452=ZZ2452+ZZ2452
+007120	121400		ZZ2452=ZZ2452+ZZ2452
+007120	010030		0 8192 -ZZ1452
+007121	121400		0 ZZ2452
-	 mark 4097, 211		/ 9 virg
+007122	000646		ZZ2453=ZZ2453+ZZ2453
+007122	001514		ZZ2453=ZZ2453+ZZ2453
+007122	003230		ZZ2453=ZZ2453+ZZ2453
+007122	006460		ZZ2453=ZZ2453+ZZ2453
+007122	015140		ZZ2453=ZZ2453+ZZ2453
+007122	032300		ZZ2453=ZZ2453+ZZ2453
+007122	064600		ZZ2453=ZZ2453+ZZ2453
+007122	151400		ZZ2453=ZZ2453+ZZ2453
+007122	007777		0 8192 -ZZ1453
+007123	151400		0 ZZ2453
-	 mark 4180, -3		/15 virg
+007124	777770		ZZ2454=ZZ2454+ZZ2454
+007124	777760		ZZ2454=ZZ2454+ZZ2454
+007124	777740		ZZ2454=ZZ2454+ZZ2454
+007124	777700		ZZ2454=ZZ2454+ZZ2454
+007124	777600		ZZ2454=ZZ2454+ZZ2454
+007124	777400		ZZ2454=ZZ2454+ZZ2454
+007124	777000		ZZ2454=ZZ2454+ZZ2454
+007124	776000		ZZ2454=ZZ2454+ZZ2454
+007124	007654		0 8192 -ZZ1454
+007125	776000		0 ZZ2454
-	 mark 4185, 418		/11 coma
+007126	001504		ZZ2455=ZZ2455+ZZ2455
+007126	003210		ZZ2455=ZZ2455+ZZ2455
+007126	006420		ZZ2455=ZZ2455+ZZ2455
+007126	015040		ZZ2455=ZZ2455+ZZ2455
+007126	032100		ZZ2455=ZZ2455+ZZ2455
+007126	064200		ZZ2455=ZZ2455+ZZ2455
+007126	150400		ZZ2455=ZZ2455+ZZ2455
+007126	321000		ZZ2455=ZZ2455+ZZ2455
+007126	007647		0 8192 -ZZ1455
+007127	321000		0 ZZ2455
-	 mark 4249, -356	/ 8 corv
+007130	776466		ZZ2456=ZZ2456+ZZ2456
+007130	775154		ZZ2456=ZZ2456+ZZ2456
+007130	772330		ZZ2456=ZZ2456+ZZ2456
+007130	764660		ZZ2456=ZZ2456+ZZ2456
+007130	751540		ZZ2456=ZZ2456+ZZ2456
+007130	723300		ZZ2456=ZZ2456+ZZ2456
+007130	646600		ZZ2456=ZZ2456+ZZ2456
+007130	515400		ZZ2456=ZZ2456+ZZ2456
+007130	007547		0 8192 -ZZ1456
+007131	515400		0 ZZ2456
-	 mark 4290, -170	/26 virg
+007132	777252		ZZ2457=ZZ2457+ZZ2457
+007132	776524		ZZ2457=ZZ2457+ZZ2457
+007132	775250		ZZ2457=ZZ2457+ZZ2457
+007132	772520		ZZ2457=ZZ2457+ZZ2457
+007132	765240		ZZ2457=ZZ2457+ZZ2457
+007132	752500		ZZ2457=ZZ2457+ZZ2457
+007132	725200		ZZ2457=ZZ2457+ZZ2457
+007132	652400		ZZ2457=ZZ2457+ZZ2457
+007132	007476		0 8192 -ZZ1457
+007133	652400		0 ZZ2457
-	 mark 4305, 245		/30 virg
+007134	000752		ZZ2458=ZZ2458+ZZ2458
+007134	001724		ZZ2458=ZZ2458+ZZ2458
+007134	003650		ZZ2458=ZZ2458+ZZ2458
+007134	007520		ZZ2458=ZZ2458+ZZ2458
+007134	017240		ZZ2458=ZZ2458+ZZ2458
+007134	036500		ZZ2458=ZZ2458+ZZ2458
+007134	075200		ZZ2458=ZZ2458+ZZ2458
+007134	172400		ZZ2458=ZZ2458+ZZ2458
+007134	007457		0 8192 -ZZ1458
+007135	172400		0 ZZ2458
-	 mark 4376, -205	/40 virg
+007136	777144		ZZ2459=ZZ2459+ZZ2459
+007136	776310		ZZ2459=ZZ2459+ZZ2459
+007136	774620		ZZ2459=ZZ2459+ZZ2459
+007136	771440		ZZ2459=ZZ2459+ZZ2459
+007136	763100		ZZ2459=ZZ2459+ZZ2459
+007136	746200		ZZ2459=ZZ2459+ZZ2459
+007136	714400		ZZ2459=ZZ2459+ZZ2459
+007136	631000		ZZ2459=ZZ2459+ZZ2459
+007136	007350		0 8192 -ZZ1459
+007137	631000		0 ZZ2459
-	 mark 4403, 409		/36 coma
+007140	001462		ZZ2460=ZZ2460+ZZ2460
+007140	003144		ZZ2460=ZZ2460+ZZ2460
+007140	006310		ZZ2460=ZZ2460+ZZ2460
+007140	014620		ZZ2460=ZZ2460+ZZ2460
+007140	031440		ZZ2460=ZZ2460+ZZ2460
+007140	063100		ZZ2460=ZZ2460+ZZ2460
+007140	146200		ZZ2460=ZZ2460+ZZ2460
+007140	314400		ZZ2460=ZZ2460+ZZ2460
+007140	007315		0 8192 -ZZ1460
+007141	314400		0 ZZ2460
-	 mark 4465, -114	/51 virg
+007142	777432		ZZ2461=ZZ2461+ZZ2461
+007142	777064		ZZ2461=ZZ2461+ZZ2461
+007142	776150		ZZ2461=ZZ2461+ZZ2461
+007142	774320		ZZ2461=ZZ2461+ZZ2461
+007142	770640		ZZ2461=ZZ2461+ZZ2461
+007142	761500		ZZ2461=ZZ2461+ZZ2461
+007142	743200		ZZ2461=ZZ2461+ZZ2461
+007142	706400		ZZ2461=ZZ2461+ZZ2461
+007142	007217		0 8192 -ZZ1461
+007143	706400		0 ZZ2461
-	 mark 4466, 411		/42 coma
+007144	001466		ZZ2462=ZZ2462+ZZ2462
+007144	003154		ZZ2462=ZZ2462+ZZ2462
+007144	006330		ZZ2462=ZZ2462+ZZ2462
+007144	014660		ZZ2462=ZZ2462+ZZ2462
+007144	031540		ZZ2462=ZZ2462+ZZ2462
+007144	063300		ZZ2462=ZZ2462+ZZ2462
+007144	146600		ZZ2462=ZZ2462+ZZ2462
+007144	315400		ZZ2462=ZZ2462+ZZ2462
+007144	007216		0 8192 -ZZ1462
+007145	315400		0 ZZ2462
-	 mark 4512, -404	/61 virg
+007146	776326		ZZ2463=ZZ2463+ZZ2463
+007146	774654		ZZ2463=ZZ2463+ZZ2463
+007146	771530		ZZ2463=ZZ2463+ZZ2463
+007146	763260		ZZ2463=ZZ2463+ZZ2463
+007146	746540		ZZ2463=ZZ2463+ZZ2463
+007146	715300		ZZ2463=ZZ2463+ZZ2463
+007146	632600		ZZ2463=ZZ2463+ZZ2463
+007146	465400		ZZ2463=ZZ2463+ZZ2463
+007146	007140		0 8192 -ZZ1463
+007147	465400		0 ZZ2463
-	 mark 4563, -352	/69 virg
+007150	776476		ZZ2464=ZZ2464+ZZ2464
+007150	775174		ZZ2464=ZZ2464+ZZ2464
+007150	772370		ZZ2464=ZZ2464+ZZ2464
+007150	764760		ZZ2464=ZZ2464+ZZ2464
+007150	751740		ZZ2464=ZZ2464+ZZ2464
+007150	723700		ZZ2464=ZZ2464+ZZ2464
+007150	647600		ZZ2464=ZZ2464+ZZ2464
+007150	517400		ZZ2464=ZZ2464+ZZ2464
+007150	007055		0 8192 -ZZ1464
+007151	517400		0 ZZ2464
-	 mark 4590, -131	/74 virg
+007152	777370		ZZ2465=ZZ2465+ZZ2465
+007152	776760		ZZ2465=ZZ2465+ZZ2465
+007152	775740		ZZ2465=ZZ2465+ZZ2465
+007152	773700		ZZ2465=ZZ2465+ZZ2465
+007152	767600		ZZ2465=ZZ2465+ZZ2465
+007152	757400		ZZ2465=ZZ2465+ZZ2465
+007152	737000		ZZ2465=ZZ2465+ZZ2465
+007152	676000		ZZ2465=ZZ2465+ZZ2465
+007152	007022		0 8192 -ZZ1465
+007153	676000		0 ZZ2465
-	 mark 4603, 95		/78 virg
+007154	000276		ZZ2466=ZZ2466+ZZ2466
+007154	000574		ZZ2466=ZZ2466+ZZ2466
+007154	001370		ZZ2466=ZZ2466+ZZ2466
+007154	002760		ZZ2466=ZZ2466+ZZ2466
+007154	005740		ZZ2466=ZZ2466+ZZ2466
+007154	013700		ZZ2466=ZZ2466+ZZ2466
+007154	027600		ZZ2466=ZZ2466+ZZ2466
+007154	057400		ZZ2466=ZZ2466+ZZ2466
+007154	007005		0 8192 -ZZ1466
+007155	057400		0 ZZ2466
-	 mark 4679, 409		/ 4 boot
+007156	001462		ZZ2467=ZZ2467+ZZ2467
+007156	003144		ZZ2467=ZZ2467+ZZ2467
+007156	006310		ZZ2467=ZZ2467+ZZ2467
+007156	014620		ZZ2467=ZZ2467+ZZ2467
+007156	031440		ZZ2467=ZZ2467+ZZ2467
+007156	063100		ZZ2467=ZZ2467+ZZ2467
+007156	146200		ZZ2467=ZZ2467+ZZ2467
+007156	314400		ZZ2467=ZZ2467+ZZ2467
+007156	006671		0 8192 -ZZ1467
+007157	314400		0 ZZ2467
-	 mark 4691, 371		/ 5 boot
+007160	001346		ZZ2468=ZZ2468+ZZ2468
+007160	002714		ZZ2468=ZZ2468+ZZ2468
+007160	005630		ZZ2468=ZZ2468+ZZ2468
+007160	013460		ZZ2468=ZZ2468+ZZ2468
+007160	027140		ZZ2468=ZZ2468+ZZ2468
+007160	056300		ZZ2468=ZZ2468+ZZ2468
+007160	134600		ZZ2468=ZZ2468+ZZ2468
+007160	271400		ZZ2468=ZZ2468+ZZ2468
+007160	006655		0 8192 -ZZ1468
+007161	271400		0 ZZ2468
-	 mark 4759, 46		/93 virg
+007162	000134		ZZ2469=ZZ2469+ZZ2469
+007162	000270		ZZ2469=ZZ2469+ZZ2469
+007162	000560		ZZ2469=ZZ2469+ZZ2469
+007162	001340		ZZ2469=ZZ2469+ZZ2469
+007162	002700		ZZ2469=ZZ2469+ZZ2469
+007162	005600		ZZ2469=ZZ2469+ZZ2469
+007162	013400		ZZ2469=ZZ2469+ZZ2469
+007162	027000		ZZ2469=ZZ2469+ZZ2469
+007162	006551		0 8192 -ZZ1469
+007163	027000		0 ZZ2469
-	 mark 4820, 66		/
+007164	000204		ZZ2470=ZZ2470+ZZ2470
+007164	000410		ZZ2470=ZZ2470+ZZ2470
+007164	001020		ZZ2470=ZZ2470+ZZ2470
+007164	002040		ZZ2470=ZZ2470+ZZ2470
+007164	004100		ZZ2470=ZZ2470+ZZ2470
+007164	010200		ZZ2470=ZZ2470+ZZ2470
+007164	020400		ZZ2470=ZZ2470+ZZ2470
+007164	041000		ZZ2470=ZZ2470+ZZ2470
+007164	006454		0 8192 -ZZ1470
+007165	041000		0 ZZ2470
-	 mark 4822, -223	/98 virg
+007166	777100		ZZ2471=ZZ2471+ZZ2471
+007166	776200		ZZ2471=ZZ2471+ZZ2471
+007166	774400		ZZ2471=ZZ2471+ZZ2471
+007166	771000		ZZ2471=ZZ2471+ZZ2471
+007166	762000		ZZ2471=ZZ2471+ZZ2471
+007166	744000		ZZ2471=ZZ2471+ZZ2471
+007166	710000		ZZ2471=ZZ2471+ZZ2471
+007166	620000		ZZ2471=ZZ2471+ZZ2471
+007166	006452		0 8192 -ZZ1471
+007167	620000		0 ZZ2471
-	 mark 4840, -126	/99 virg
+007170	777402		ZZ2472=ZZ2472+ZZ2472
+007170	777004		ZZ2472=ZZ2472+ZZ2472
+007170	776010		ZZ2472=ZZ2472+ZZ2472
+007170	774020		ZZ2472=ZZ2472+ZZ2472
+007170	770040		ZZ2472=ZZ2472+ZZ2472
+007170	760100		ZZ2472=ZZ2472+ZZ2472
+007170	740200		ZZ2472=ZZ2472+ZZ2472
+007170	700400		ZZ2472=ZZ2472+ZZ2472
+007170	006430		0 8192 -ZZ1472
+007171	700400		0 ZZ2472
-	 mark 4857, -294	/100 virg
+007172	776662		ZZ2473=ZZ2473+ZZ2473
+007172	775544		ZZ2473=ZZ2473+ZZ2473
+007172	773310		ZZ2473=ZZ2473+ZZ2473
+007172	766620		ZZ2473=ZZ2473+ZZ2473
+007172	755440		ZZ2473=ZZ2473+ZZ2473
+007172	733100		ZZ2473=ZZ2473+ZZ2473
+007172	666200		ZZ2473=ZZ2473+ZZ2473
+007172	554400		ZZ2473=ZZ2473+ZZ2473
+007172	006407		0 8192 -ZZ1473
+007173	554400		0 ZZ2473
-	 mark 4864, 382		/20 boot
+007174	001374		ZZ2474=ZZ2474+ZZ2474
+007174	002770		ZZ2474=ZZ2474+ZZ2474
+007174	005760		ZZ2474=ZZ2474+ZZ2474
+007174	013740		ZZ2474=ZZ2474+ZZ2474
+007174	027700		ZZ2474=ZZ2474+ZZ2474
+007174	057600		ZZ2474=ZZ2474+ZZ2474
+007174	137400		ZZ2474=ZZ2474+ZZ2474
+007174	277000		ZZ2474=ZZ2474+ZZ2474
+007174	006400		0 8192 -ZZ1474
+007175	277000		0 ZZ2474
-	 mark 4910, -41		/105 virg
+007176	777654		ZZ2475=ZZ2475+ZZ2475
+007176	777530		ZZ2475=ZZ2475+ZZ2475
+007176	777260		ZZ2475=ZZ2475+ZZ2475
+007176	776540		ZZ2475=ZZ2475+ZZ2475
+007176	775300		ZZ2475=ZZ2475+ZZ2475
+007176	772600		ZZ2475=ZZ2475+ZZ2475
+007176	765400		ZZ2475=ZZ2475+ZZ2475
+007176	753000		ZZ2475=ZZ2475+ZZ2475
+007176	006322		0 8192 -ZZ1475
+007177	753000		0 ZZ2475
-	 mark 4984, 383		/29 boot
+007200	001376		ZZ2476=ZZ2476+ZZ2476
+007200	002774		ZZ2476=ZZ2476+ZZ2476
+007200	005770		ZZ2476=ZZ2476+ZZ2476
+007200	013760		ZZ2476=ZZ2476+ZZ2476
+007200	027740		ZZ2476=ZZ2476+ZZ2476
+007200	057700		ZZ2476=ZZ2476+ZZ2476
+007200	137600		ZZ2476=ZZ2476+ZZ2476
+007200	277400		ZZ2476=ZZ2476+ZZ2476
+007200	006210		0 8192 -ZZ1476
+007201	277400		0 ZZ2476
-	 mark 4986, 322		/30 boot
+007202	001204		ZZ2477=ZZ2477+ZZ2477
+007202	002410		ZZ2477=ZZ2477+ZZ2477
+007202	005020		ZZ2477=ZZ2477+ZZ2477
+007202	012040		ZZ2477=ZZ2477+ZZ2477
+007202	024100		ZZ2477=ZZ2477+ZZ2477
+007202	050200		ZZ2477=ZZ2477+ZZ2477
+007202	120400		ZZ2477=ZZ2477+ZZ2477
+007202	241000		ZZ2477=ZZ2477+ZZ2477
+007202	006206		0 8192 -ZZ1477
+007203	241000		0 ZZ2477
-	 mark 4994, -119	/107 virg
+007204	777420		ZZ2478=ZZ2478+ZZ2478
+007204	777040		ZZ2478=ZZ2478+ZZ2478
+007204	776100		ZZ2478=ZZ2478+ZZ2478
+007204	774200		ZZ2478=ZZ2478+ZZ2478
+007204	770400		ZZ2478=ZZ2478+ZZ2478
+007204	761000		ZZ2478=ZZ2478+ZZ2478
+007204	742000		ZZ2478=ZZ2478+ZZ2478
+007204	704000		ZZ2478=ZZ2478+ZZ2478
+007204	006176		0 8192 -ZZ1478
+007205	704000		0 ZZ2478
-	 mark 5009, 396		/35 boot
+007206	001430		ZZ2479=ZZ2479+ZZ2479
+007206	003060		ZZ2479=ZZ2479+ZZ2479
+007206	006140		ZZ2479=ZZ2479+ZZ2479
+007206	014300		ZZ2479=ZZ2479+ZZ2479
+007206	030600		ZZ2479=ZZ2479+ZZ2479
+007206	061400		ZZ2479=ZZ2479+ZZ2479
+007206	143000		ZZ2479=ZZ2479+ZZ2479
+007206	306000		ZZ2479=ZZ2479+ZZ2479
+007206	006157		0 8192 -ZZ1479
+007207	306000		0 ZZ2479
-	 mark 5013, 53		/109 virg
+007210	000152		ZZ2480=ZZ2480+ZZ2480
+007210	000324		ZZ2480=ZZ2480+ZZ2480
+007210	000650		ZZ2480=ZZ2480+ZZ2480
+007210	001520		ZZ2480=ZZ2480+ZZ2480
+007210	003240		ZZ2480=ZZ2480+ZZ2480
+007210	006500		ZZ2480=ZZ2480+ZZ2480
+007210	015200		ZZ2480=ZZ2480+ZZ2480
+007210	032400		ZZ2480=ZZ2480+ZZ2480
+007210	006153		0 8192 -ZZ1480
+007211	032400		0 ZZ2480
-	 mark 5045, 444		/37 boot
+007212	001570		ZZ2481=ZZ2481+ZZ2481
+007212	003360		ZZ2481=ZZ2481+ZZ2481
+007212	006740		ZZ2481=ZZ2481+ZZ2481
+007212	015700		ZZ2481=ZZ2481+ZZ2481
+007212	033600		ZZ2481=ZZ2481+ZZ2481
+007212	067400		ZZ2481=ZZ2481+ZZ2481
+007212	157000		ZZ2481=ZZ2481+ZZ2481
+007212	336000		ZZ2481=ZZ2481+ZZ2481
+007212	006113		0 8192 -ZZ1481
+007213	336000		0 ZZ2481
-	 mark 5074, -90		/16 libr
+007214	777512		ZZ2482=ZZ2482+ZZ2482
+007214	777224		ZZ2482=ZZ2482+ZZ2482
+007214	776450		ZZ2482=ZZ2482+ZZ2482
+007214	775120		ZZ2482=ZZ2482+ZZ2482
+007214	772240		ZZ2482=ZZ2482+ZZ2482
+007214	764500		ZZ2482=ZZ2482+ZZ2482
+007214	751200		ZZ2482=ZZ2482+ZZ2482
+007214	722400		ZZ2482=ZZ2482+ZZ2482
+007214	006056		0 8192 -ZZ1482
+007215	722400		0 ZZ2482
-	 mark 5108, 57		/110 virg
+007216	000162		ZZ2483=ZZ2483+ZZ2483
+007216	000344		ZZ2483=ZZ2483+ZZ2483
+007216	000710		ZZ2483=ZZ2483+ZZ2483
+007216	001620		ZZ2483=ZZ2483+ZZ2483
+007216	003440		ZZ2483=ZZ2483+ZZ2483
+007216	007100		ZZ2483=ZZ2483+ZZ2483
+007216	016200		ZZ2483=ZZ2483+ZZ2483
+007216	034400		ZZ2483=ZZ2483+ZZ2483
+007216	006014		0 8192 -ZZ1483
+007217	034400		0 ZZ2483
-	 mark 5157, -442	/24 libr
+007220	776212		ZZ2484=ZZ2484+ZZ2484
+007220	774424		ZZ2484=ZZ2484+ZZ2484
+007220	771050		ZZ2484=ZZ2484+ZZ2484
+007220	762120		ZZ2484=ZZ2484+ZZ2484
+007220	744240		ZZ2484=ZZ2484+ZZ2484
+007220	710500		ZZ2484=ZZ2484+ZZ2484
+007220	621200		ZZ2484=ZZ2484+ZZ2484
+007220	442400		ZZ2484=ZZ2484+ZZ2484
+007220	005733		0 8192 -ZZ1484
+007221	442400		0 ZZ2484
-	 mark 5283, -221	/37 libr
+007222	777104		ZZ2485=ZZ2485+ZZ2485
+007222	776210		ZZ2485=ZZ2485+ZZ2485
+007222	774420		ZZ2485=ZZ2485+ZZ2485
+007222	771040		ZZ2485=ZZ2485+ZZ2485
+007222	762100		ZZ2485=ZZ2485+ZZ2485
+007222	744200		ZZ2485=ZZ2485+ZZ2485
+007222	710400		ZZ2485=ZZ2485+ZZ2485
+007222	621000		ZZ2485=ZZ2485+ZZ2485
+007222	005535		0 8192 -ZZ1485
+007223	621000		0 ZZ2485
-	 mark 5290, -329	/38 libr
+007224	776554		ZZ2486=ZZ2486+ZZ2486
+007224	775330		ZZ2486=ZZ2486+ZZ2486
+007224	772660		ZZ2486=ZZ2486+ZZ2486
+007224	765540		ZZ2486=ZZ2486+ZZ2486
+007224	753300		ZZ2486=ZZ2486+ZZ2486
+007224	726600		ZZ2486=ZZ2486+ZZ2486
+007224	655400		ZZ2486=ZZ2486+ZZ2486
+007224	533000		ZZ2486=ZZ2486+ZZ2486
+007224	005526		0 8192 -ZZ1486
+007225	533000		0 ZZ2486
-	 mark 5291, 247		/13 serp
+007226	000756		ZZ2487=ZZ2487+ZZ2487
+007226	001734		ZZ2487=ZZ2487+ZZ2487
+007226	003670		ZZ2487=ZZ2487+ZZ2487
+007226	007560		ZZ2487=ZZ2487+ZZ2487
+007226	017340		ZZ2487=ZZ2487+ZZ2487
+007226	036700		ZZ2487=ZZ2487+ZZ2487
+007226	075600		ZZ2487=ZZ2487+ZZ2487
+007226	173400		ZZ2487=ZZ2487+ZZ2487
+007226	005525		0 8192 -ZZ1487
+007227	173400		0 ZZ2487
-	 mark 5326, -440	/43 libr
+007230	776216		ZZ2488=ZZ2488+ZZ2488
+007230	774434		ZZ2488=ZZ2488+ZZ2488
+007230	771070		ZZ2488=ZZ2488+ZZ2488
+007230	762160		ZZ2488=ZZ2488+ZZ2488
+007230	744340		ZZ2488=ZZ2488+ZZ2488
+007230	710700		ZZ2488=ZZ2488+ZZ2488
+007230	621600		ZZ2488=ZZ2488+ZZ2488
+007230	443400		ZZ2488=ZZ2488+ZZ2488
+007230	005462		0 8192 -ZZ1488
+007231	443400		0 ZZ2488
-	 mark 5331, 455		/21 serp
+007232	001616		ZZ2489=ZZ2489+ZZ2489
+007232	003434		ZZ2489=ZZ2489+ZZ2489
+007232	007070		ZZ2489=ZZ2489+ZZ2489
+007232	016160		ZZ2489=ZZ2489+ZZ2489
+007232	034340		ZZ2489=ZZ2489+ZZ2489
+007232	070700		ZZ2489=ZZ2489+ZZ2489
+007232	161600		ZZ2489=ZZ2489+ZZ2489
+007232	343400		ZZ2489=ZZ2489+ZZ2489
+007232	005455		0 8192 -ZZ1489
+007233	343400		0 ZZ2489
-	 mark 5357, 175		/27 serp
+007234	000536		ZZ2490=ZZ2490+ZZ2490
+007234	001274		ZZ2490=ZZ2490+ZZ2490
+007234	002570		ZZ2490=ZZ2490+ZZ2490
+007234	005360		ZZ2490=ZZ2490+ZZ2490
+007234	012740		ZZ2490=ZZ2490+ZZ2490
+007234	025700		ZZ2490=ZZ2490+ZZ2490
+007234	053600		ZZ2490=ZZ2490+ZZ2490
+007234	127400		ZZ2490=ZZ2490+ZZ2490
+007234	005423		0 8192 -ZZ1490
+007235	127400		0 ZZ2490
-	 mark 5372, 420		/35 serp
+007236	001510		ZZ2491=ZZ2491+ZZ2491
+007236	003220		ZZ2491=ZZ2491+ZZ2491
+007236	006440		ZZ2491=ZZ2491+ZZ2491
+007236	015100		ZZ2491=ZZ2491+ZZ2491
+007236	032200		ZZ2491=ZZ2491+ZZ2491
+007236	064400		ZZ2491=ZZ2491+ZZ2491
+007236	151000		ZZ2491=ZZ2491+ZZ2491
+007236	322000		ZZ2491=ZZ2491+ZZ2491
+007236	005404		0 8192 -ZZ1491
+007237	322000		0 ZZ2491
-	 mark 5381, 109		/37 serp
+007240	000332		ZZ2492=ZZ2492+ZZ2492
+007240	000664		ZZ2492=ZZ2492+ZZ2492
+007240	001550		ZZ2492=ZZ2492+ZZ2492
+007240	003320		ZZ2492=ZZ2492+ZZ2492
+007240	006640		ZZ2492=ZZ2492+ZZ2492
+007240	015500		ZZ2492=ZZ2492+ZZ2492
+007240	033200		ZZ2492=ZZ2492+ZZ2492
+007240	066400		ZZ2492=ZZ2492+ZZ2492
+007240	005373		0 8192 -ZZ1492
+007241	066400		0 ZZ2492
-	 mark 5387, 484		/38 serp
+007242	001710		ZZ2493=ZZ2493+ZZ2493
+007242	003620		ZZ2493=ZZ2493+ZZ2493
+007242	007440		ZZ2493=ZZ2493+ZZ2493
+007242	017100		ZZ2493=ZZ2493+ZZ2493
+007242	036200		ZZ2493=ZZ2493+ZZ2493
+007242	074400		ZZ2493=ZZ2493+ZZ2493
+007242	171000		ZZ2493=ZZ2493+ZZ2493
+007242	362000		ZZ2493=ZZ2493+ZZ2493
+007242	005365		0 8192 -ZZ1493
+007243	362000		0 ZZ2493
-	 mark 5394, -374	/46 libr
+007244	776422		ZZ2494=ZZ2494+ZZ2494
+007244	775044		ZZ2494=ZZ2494+ZZ2494
+007244	772110		ZZ2494=ZZ2494+ZZ2494
+007244	764220		ZZ2494=ZZ2494+ZZ2494
+007244	750440		ZZ2494=ZZ2494+ZZ2494
+007244	721100		ZZ2494=ZZ2494+ZZ2494
+007244	642200		ZZ2494=ZZ2494+ZZ2494
+007244	504400		ZZ2494=ZZ2494+ZZ2494
+007244	005356		0 8192 -ZZ1494
+007245	504400		0 ZZ2494
-	 mark 5415, 364		/41 serp
+007246	001330		ZZ2495=ZZ2495+ZZ2495
+007246	002660		ZZ2495=ZZ2495+ZZ2495
+007246	005540		ZZ2495=ZZ2495+ZZ2495
+007246	013300		ZZ2495=ZZ2495+ZZ2495
+007246	026600		ZZ2495=ZZ2495+ZZ2495
+007246	055400		ZZ2495=ZZ2495+ZZ2495
+007246	133000		ZZ2495=ZZ2495+ZZ2495
+007246	266000		ZZ2495=ZZ2495+ZZ2495
+007246	005331		0 8192 -ZZ1495
+007247	266000		0 ZZ2495
-	 mark 5419, -318	/48 libr
+007250	776602		ZZ2496=ZZ2496+ZZ2496
+007250	775404		ZZ2496=ZZ2496+ZZ2496
+007250	773010		ZZ2496=ZZ2496+ZZ2496
+007250	766020		ZZ2496=ZZ2496+ZZ2496
+007250	754040		ZZ2496=ZZ2496+ZZ2496
+007250	730100		ZZ2496=ZZ2496+ZZ2496
+007250	660200		ZZ2496=ZZ2496+ZZ2496
+007250	540400		ZZ2496=ZZ2496+ZZ2496
+007250	005325		0 8192 -ZZ1496
+007251	540400		0 ZZ2496
-	 mark 5455, -253	/xi scor
+007252	777004		ZZ2497=ZZ2497+ZZ2497
+007252	776010		ZZ2497=ZZ2497+ZZ2497
+007252	774020		ZZ2497=ZZ2497+ZZ2497
+007252	770040		ZZ2497=ZZ2497+ZZ2497
+007252	760100		ZZ2497=ZZ2497+ZZ2497
+007252	740200		ZZ2497=ZZ2497+ZZ2497
+007252	700400		ZZ2497=ZZ2497+ZZ2497
+007252	601000		ZZ2497=ZZ2497+ZZ2497
+007252	005261		0 8192 -ZZ1497
+007253	601000		0 ZZ2497
-	 mark 5467, -464	/ 9 scor
+007254	776136		ZZ2498=ZZ2498+ZZ2498
+007254	774274		ZZ2498=ZZ2498+ZZ2498
+007254	770570		ZZ2498=ZZ2498+ZZ2498
+007254	761360		ZZ2498=ZZ2498+ZZ2498
+007254	742740		ZZ2498=ZZ2498+ZZ2498
+007254	705700		ZZ2498=ZZ2498+ZZ2498
+007254	613600		ZZ2498=ZZ2498+ZZ2498
+007254	427400		ZZ2498=ZZ2498+ZZ2498
+007254	005245		0 8192 -ZZ1498
+007255	427400		0 ZZ2498
-	 mark 5470, -469	/10 scor
+007256	776124		ZZ2499=ZZ2499+ZZ2499
+007256	774250		ZZ2499=ZZ2499+ZZ2499
+007256	770520		ZZ2499=ZZ2499+ZZ2499
+007256	761240		ZZ2499=ZZ2499+ZZ2499
+007256	742500		ZZ2499=ZZ2499+ZZ2499
+007256	705200		ZZ2499=ZZ2499+ZZ2499
+007256	612400		ZZ2499=ZZ2499+ZZ2499
+007256	425000		ZZ2499=ZZ2499+ZZ2499
+007256	005242		0 8192 -ZZ1499
+007257	425000		0 ZZ2499
-	 mark 5497, -437	/14 scor
+007260	776224		ZZ2500=ZZ2500+ZZ2500
+007260	774450		ZZ2500=ZZ2500+ZZ2500
+007260	771120		ZZ2500=ZZ2500+ZZ2500
+007260	762240		ZZ2500=ZZ2500+ZZ2500
+007260	744500		ZZ2500=ZZ2500+ZZ2500
+007260	711200		ZZ2500=ZZ2500+ZZ2500
+007260	622400		ZZ2500=ZZ2500+ZZ2500
+007260	445000		ZZ2500=ZZ2500+ZZ2500
+007260	005207		0 8192 -ZZ1500
+007261	445000		0 ZZ2500
-	 mark 5499, -223	/15 scor
+007262	777100		ZZ2501=ZZ2501+ZZ2501
+007262	776200		ZZ2501=ZZ2501+ZZ2501
+007262	774400		ZZ2501=ZZ2501+ZZ2501
+007262	771000		ZZ2501=ZZ2501+ZZ2501
+007262	762000		ZZ2501=ZZ2501+ZZ2501
+007262	744000		ZZ2501=ZZ2501+ZZ2501
+007262	710000		ZZ2501=ZZ2501+ZZ2501
+007262	620000		ZZ2501=ZZ2501+ZZ2501
+007262	005205		0 8192 -ZZ1501
+007263	620000		0 ZZ2501
-	 mark 5558, 29		/50 serp
+007264	000072		ZZ2502=ZZ2502+ZZ2502
+007264	000164		ZZ2502=ZZ2502+ZZ2502
+007264	000350		ZZ2502=ZZ2502+ZZ2502
+007264	000720		ZZ2502=ZZ2502+ZZ2502
+007264	001640		ZZ2502=ZZ2502+ZZ2502
+007264	003500		ZZ2502=ZZ2502+ZZ2502
+007264	007200		ZZ2502=ZZ2502+ZZ2502
+007264	016400		ZZ2502=ZZ2502+ZZ2502
+007264	005112		0 8192 -ZZ1502
+007265	016400		0 ZZ2502
-	 mark 5561, 441		/20 herc
+007266	001562		ZZ2503=ZZ2503+ZZ2503
+007266	003344		ZZ2503=ZZ2503+ZZ2503
+007266	006710		ZZ2503=ZZ2503+ZZ2503
+007266	015620		ZZ2503=ZZ2503+ZZ2503
+007266	033440		ZZ2503=ZZ2503+ZZ2503
+007266	067100		ZZ2503=ZZ2503+ZZ2503
+007266	156200		ZZ2503=ZZ2503+ZZ2503
+007266	334400		ZZ2503=ZZ2503+ZZ2503
+007266	005107		0 8192 -ZZ1503
+007267	334400		0 ZZ2503
-	 mark 5565, -451	/ 4 ophi
+007270	776170		ZZ2504=ZZ2504+ZZ2504
+007270	774360		ZZ2504=ZZ2504+ZZ2504
+007270	770740		ZZ2504=ZZ2504+ZZ2504
+007270	761700		ZZ2504=ZZ2504+ZZ2504
+007270	743600		ZZ2504=ZZ2504+ZZ2504
+007270	707400		ZZ2504=ZZ2504+ZZ2504
+007270	617000		ZZ2504=ZZ2504+ZZ2504
+007270	436000		ZZ2504=ZZ2504+ZZ2504
+007270	005103		0 8192 -ZZ1504
+007271	436000		0 ZZ2504
-	 mark 5580, 325		/24 herc
+007272	001212		ZZ2505=ZZ2505+ZZ2505
+007272	002424		ZZ2505=ZZ2505+ZZ2505
+007272	005050		ZZ2505=ZZ2505+ZZ2505
+007272	012120		ZZ2505=ZZ2505+ZZ2505
+007272	024240		ZZ2505=ZZ2505+ZZ2505
+007272	050500		ZZ2505=ZZ2505+ZZ2505
+007272	121200		ZZ2505=ZZ2505+ZZ2505
+007272	242400		ZZ2505=ZZ2505+ZZ2505
+007272	005064		0 8192 -ZZ1505
+007273	242400		0 ZZ2505
-	 mark 5582, -415	/ 7 ophi
+007274	776300		ZZ2506=ZZ2506+ZZ2506
+007274	774600		ZZ2506=ZZ2506+ZZ2506
+007274	771400		ZZ2506=ZZ2506+ZZ2506
+007274	763000		ZZ2506=ZZ2506+ZZ2506
+007274	746000		ZZ2506=ZZ2506+ZZ2506
+007274	714000		ZZ2506=ZZ2506+ZZ2506
+007274	630000		ZZ2506=ZZ2506+ZZ2506
+007274	460000		ZZ2506=ZZ2506+ZZ2506
+007274	005062		0 8192 -ZZ1506
+007275	460000		0 ZZ2506
-	 mark 5589, -186	/ 3 ophi
+007276	777212		ZZ2507=ZZ2507+ZZ2507
+007276	776424		ZZ2507=ZZ2507+ZZ2507
+007276	775050		ZZ2507=ZZ2507+ZZ2507
+007276	772120		ZZ2507=ZZ2507+ZZ2507
+007276	764240		ZZ2507=ZZ2507+ZZ2507
+007276	750500		ZZ2507=ZZ2507+ZZ2507
+007276	721200		ZZ2507=ZZ2507+ZZ2507
+007276	642400		ZZ2507=ZZ2507+ZZ2507
+007276	005053		0 8192 -ZZ1507
+007277	642400		0 ZZ2507
-	 mark 5606, -373	/ 8 ophi
+007300	776424		ZZ2508=ZZ2508+ZZ2508
+007300	775050		ZZ2508=ZZ2508+ZZ2508
+007300	772120		ZZ2508=ZZ2508+ZZ2508
+007300	764240		ZZ2508=ZZ2508+ZZ2508
+007300	750500		ZZ2508=ZZ2508+ZZ2508
+007300	721200		ZZ2508=ZZ2508+ZZ2508
+007300	642400		ZZ2508=ZZ2508+ZZ2508
+007300	505000		ZZ2508=ZZ2508+ZZ2508
+007300	005032		0 8192 -ZZ1508
+007301	505000		0 ZZ2508
-	 mark 5609, 50		/10 ophi
+007302	000144		ZZ2509=ZZ2509+ZZ2509
+007302	000310		ZZ2509=ZZ2509+ZZ2509
+007302	000620		ZZ2509=ZZ2509+ZZ2509
+007302	001440		ZZ2509=ZZ2509+ZZ2509
+007302	003100		ZZ2509=ZZ2509+ZZ2509
+007302	006200		ZZ2509=ZZ2509+ZZ2509
+007302	014400		ZZ2509=ZZ2509+ZZ2509
+007302	031000		ZZ2509=ZZ2509+ZZ2509
+007302	005027		0 8192 -ZZ1509
+007303	031000		0 ZZ2509
-	 mark 5610, -484	/ 9 ophi
+007304	776066		ZZ2510=ZZ2510+ZZ2510
+007304	774154		ZZ2510=ZZ2510+ZZ2510
+007304	770330		ZZ2510=ZZ2510+ZZ2510
+007304	760660		ZZ2510=ZZ2510+ZZ2510
+007304	741540		ZZ2510=ZZ2510+ZZ2510
+007304	703300		ZZ2510=ZZ2510+ZZ2510
+007304	606600		ZZ2510=ZZ2510+ZZ2510
+007304	415400		ZZ2510=ZZ2510+ZZ2510
+007304	005026		0 8192 -ZZ1510
+007305	415400		0 ZZ2510
-	 mark 5620, 266		/29 herc
+007306	001024		ZZ2511=ZZ2511+ZZ2511
+007306	002050		ZZ2511=ZZ2511+ZZ2511
+007306	004120		ZZ2511=ZZ2511+ZZ2511
+007306	010240		ZZ2511=ZZ2511+ZZ2511
+007306	020500		ZZ2511=ZZ2511+ZZ2511
+007306	041200		ZZ2511=ZZ2511+ZZ2511
+007306	102400		ZZ2511=ZZ2511+ZZ2511
+007306	205000		ZZ2511=ZZ2511+ZZ2511
+007306	005014		0 8192 -ZZ1511
+007307	205000		0 ZZ2511
-	 mark 5713, -241	/20 ophi
+007310	777034		ZZ2512=ZZ2512+ZZ2512
+007310	776070		ZZ2512=ZZ2512+ZZ2512
+007310	774160		ZZ2512=ZZ2512+ZZ2512
+007310	770340		ZZ2512=ZZ2512+ZZ2512
+007310	760700		ZZ2512=ZZ2512+ZZ2512
+007310	741600		ZZ2512=ZZ2512+ZZ2512
+007310	703400		ZZ2512=ZZ2512+ZZ2512
+007310	607000		ZZ2512=ZZ2512+ZZ2512
+007310	004657		0 8192 -ZZ1512
+007311	607000		0 ZZ2512
-	 mark 5742, 235		/25 ophi
+007312	000726		ZZ2513=ZZ2513+ZZ2513
+007312	001654		ZZ2513=ZZ2513+ZZ2513
+007312	003530		ZZ2513=ZZ2513+ZZ2513
+007312	007260		ZZ2513=ZZ2513+ZZ2513
+007312	016540		ZZ2513=ZZ2513+ZZ2513
+007312	035300		ZZ2513=ZZ2513+ZZ2513
+007312	072600		ZZ2513=ZZ2513+ZZ2513
+007312	165400		ZZ2513=ZZ2513+ZZ2513
+007312	004622		0 8192 -ZZ1513
+007313	165400		0 ZZ2513
-	 mark 5763, 217		/27 ophi
+007314	000662		ZZ2514=ZZ2514+ZZ2514
+007314	001544		ZZ2514=ZZ2514+ZZ2514
+007314	003310		ZZ2514=ZZ2514+ZZ2514
+007314	006620		ZZ2514=ZZ2514+ZZ2514
+007314	015440		ZZ2514=ZZ2514+ZZ2514
+007314	033100		ZZ2514=ZZ2514+ZZ2514
+007314	066200		ZZ2514=ZZ2514+ZZ2514
+007314	154400		ZZ2514=ZZ2514+ZZ2514
+007314	004575		0 8192 -ZZ1514
+007315	154400		0 ZZ2514
-	 mark 5807, 293		/60 herc
+007316	001112		ZZ2515=ZZ2515+ZZ2515
+007316	002224		ZZ2515=ZZ2515+ZZ2515
+007316	004450		ZZ2515=ZZ2515+ZZ2515
+007316	011120		ZZ2515=ZZ2515+ZZ2515
+007316	022240		ZZ2515=ZZ2515+ZZ2515
+007316	044500		ZZ2515=ZZ2515+ZZ2515
+007316	111200		ZZ2515=ZZ2515+ZZ2515
+007316	222400		ZZ2515=ZZ2515+ZZ2515
+007316	004521		0 8192 -ZZ1515
+007317	222400		0 ZZ2515
-	 mark 5868, -8		/41 ophi
+007320	777756		ZZ2516=ZZ2516+ZZ2516
+007320	777734		ZZ2516=ZZ2516+ZZ2516
+007320	777670		ZZ2516=ZZ2516+ZZ2516
+007320	777560		ZZ2516=ZZ2516+ZZ2516
+007320	777340		ZZ2516=ZZ2516+ZZ2516
+007320	776700		ZZ2516=ZZ2516+ZZ2516
+007320	775600		ZZ2516=ZZ2516+ZZ2516
+007320	773400		ZZ2516=ZZ2516+ZZ2516
+007320	004424		0 8192 -ZZ1516
+007321	773400		0 ZZ2516
-	 mark 5888, -478	/40 ophi
+007322	776102		ZZ2517=ZZ2517+ZZ2517
+007322	774204		ZZ2517=ZZ2517+ZZ2517
+007322	770410		ZZ2517=ZZ2517+ZZ2517
+007322	761020		ZZ2517=ZZ2517+ZZ2517
+007322	742040		ZZ2517=ZZ2517+ZZ2517
+007322	704100		ZZ2517=ZZ2517+ZZ2517
+007322	610200		ZZ2517=ZZ2517+ZZ2517
+007322	420400		ZZ2517=ZZ2517+ZZ2517
+007322	004400		0 8192 -ZZ1517
+007323	420400		0 ZZ2517
-	 mark 5889, -290	/53 serp
+007324	776672		ZZ2518=ZZ2518+ZZ2518
+007324	775564		ZZ2518=ZZ2518+ZZ2518
+007324	773350		ZZ2518=ZZ2518+ZZ2518
+007324	766720		ZZ2518=ZZ2518+ZZ2518
+007324	755640		ZZ2518=ZZ2518+ZZ2518
+007324	733500		ZZ2518=ZZ2518+ZZ2518
+007324	667200		ZZ2518=ZZ2518+ZZ2518
+007324	556400		ZZ2518=ZZ2518+ZZ2518
+007324	004377		0 8192 -ZZ1518
+007325	556400		0 ZZ2518
-	 mark 5924, -114	/
+007326	777432		ZZ2519=ZZ2519+ZZ2519
+007326	777064		ZZ2519=ZZ2519+ZZ2519
+007326	776150		ZZ2519=ZZ2519+ZZ2519
+007326	774320		ZZ2519=ZZ2519+ZZ2519
+007326	770640		ZZ2519=ZZ2519+ZZ2519
+007326	761500		ZZ2519=ZZ2519+ZZ2519
+007326	743200		ZZ2519=ZZ2519+ZZ2519
+007326	706400		ZZ2519=ZZ2519+ZZ2519
+007326	004334		0 8192 -ZZ1519
+007327	706400		0 ZZ2519
-	 mark 5925, 96		/49 ophi
+007330	000300		ZZ2520=ZZ2520+ZZ2520
+007330	000600		ZZ2520=ZZ2520+ZZ2520
+007330	001400		ZZ2520=ZZ2520+ZZ2520
+007330	003000		ZZ2520=ZZ2520+ZZ2520
+007330	006000		ZZ2520=ZZ2520+ZZ2520
+007330	014000		ZZ2520=ZZ2520+ZZ2520
+007330	030000		ZZ2520=ZZ2520+ZZ2520
+007330	060000		ZZ2520=ZZ2520+ZZ2520
+007330	004333		0 8192 -ZZ1520
+007331	060000		0 ZZ2520
-	 mark 5987, -183	/57 ophi
+007332	777220		ZZ2521=ZZ2521+ZZ2521
+007332	776440		ZZ2521=ZZ2521+ZZ2521
+007332	775100		ZZ2521=ZZ2521+ZZ2521
+007332	772200		ZZ2521=ZZ2521+ZZ2521
+007332	764400		ZZ2521=ZZ2521+ZZ2521
+007332	751000		ZZ2521=ZZ2521+ZZ2521
+007332	722000		ZZ2521=ZZ2521+ZZ2521
+007332	644000		ZZ2521=ZZ2521+ZZ2521
+007332	004235		0 8192 -ZZ1521
+007333	644000		0 ZZ2521
-	 mark 6006, -292	/56 serp
+007334	776666		ZZ2522=ZZ2522+ZZ2522
+007334	775554		ZZ2522=ZZ2522+ZZ2522
+007334	773330		ZZ2522=ZZ2522+ZZ2522
+007334	766660		ZZ2522=ZZ2522+ZZ2522
+007334	755540		ZZ2522=ZZ2522+ZZ2522
+007334	733300		ZZ2522=ZZ2522+ZZ2522
+007334	666600		ZZ2522=ZZ2522+ZZ2522
+007334	555400		ZZ2522=ZZ2522+ZZ2522
+007334	004212		0 8192 -ZZ1522
+007335	555400		0 ZZ2522
-	 mark 6016, -492	/58 ophi
+007336	776046		ZZ2523=ZZ2523+ZZ2523
+007336	774114		ZZ2523=ZZ2523+ZZ2523
+007336	770230		ZZ2523=ZZ2523+ZZ2523
+007336	760460		ZZ2523=ZZ2523+ZZ2523
+007336	741140		ZZ2523=ZZ2523+ZZ2523
+007336	702300		ZZ2523=ZZ2523+ZZ2523
+007336	604600		ZZ2523=ZZ2523+ZZ2523
+007336	411400		ZZ2523=ZZ2523+ZZ2523
+007336	004200		0 8192 -ZZ1523
+007337	411400		0 ZZ2523
-	 mark 6117, -84		/57 serp
+007340	777526		ZZ2524=ZZ2524+ZZ2524
+007340	777254		ZZ2524=ZZ2524+ZZ2524
+007340	776530		ZZ2524=ZZ2524+ZZ2524
+007340	775260		ZZ2524=ZZ2524+ZZ2524
+007340	772540		ZZ2524=ZZ2524+ZZ2524
+007340	765300		ZZ2524=ZZ2524+ZZ2524
+007340	752600		ZZ2524=ZZ2524+ZZ2524
+007340	725400		ZZ2524=ZZ2524+ZZ2524
+007340	004033		0 8192 -ZZ1524
+007341	725400		0 ZZ2524
-	 mark 6117, 99		/66 ophi
+007342	000306		ZZ2525=ZZ2525+ZZ2525
+007342	000614		ZZ2525=ZZ2525+ZZ2525
+007342	001430		ZZ2525=ZZ2525+ZZ2525
+007342	003060		ZZ2525=ZZ2525+ZZ2525
+007342	006140		ZZ2525=ZZ2525+ZZ2525
+007342	014300		ZZ2525=ZZ2525+ZZ2525
+007342	030600		ZZ2525=ZZ2525+ZZ2525
+007342	061400		ZZ2525=ZZ2525+ZZ2525
+007342	004033		0 8192 -ZZ1525
+007343	061400		0 ZZ2525
-	 mark 6119, 381		/93 herc
+007344	001372		ZZ2526=ZZ2526+ZZ2526
+007344	002764		ZZ2526=ZZ2526+ZZ2526
+007344	005750		ZZ2526=ZZ2526+ZZ2526
+007344	013720		ZZ2526=ZZ2526+ZZ2526
+007344	027640		ZZ2526=ZZ2526+ZZ2526
+007344	057500		ZZ2526=ZZ2526+ZZ2526
+007344	137200		ZZ2526=ZZ2526+ZZ2526
+007344	276400		ZZ2526=ZZ2526+ZZ2526
+007344	004031		0 8192 -ZZ1526
+007345	276400		0 ZZ2526
-	 mark 6119, 67		/67 ophi
+007346	000206		ZZ2527=ZZ2527+ZZ2527
+007346	000414		ZZ2527=ZZ2527+ZZ2527
+007346	001030		ZZ2527=ZZ2527+ZZ2527
+007346	002060		ZZ2527=ZZ2527+ZZ2527
+007346	004140		ZZ2527=ZZ2527+ZZ2527
+007346	010300		ZZ2527=ZZ2527+ZZ2527
+007346	020600		ZZ2527=ZZ2527+ZZ2527
+007346	041400		ZZ2527=ZZ2527+ZZ2527
+007346	004031		0 8192 -ZZ1527
+007347	041400		0 ZZ2527
-	 mark 6125, 30		/68 ophi
+007350	000074		ZZ2528=ZZ2528+ZZ2528
+007350	000170		ZZ2528=ZZ2528+ZZ2528
+007350	000360		ZZ2528=ZZ2528+ZZ2528
+007350	000740		ZZ2528=ZZ2528+ZZ2528
+007350	001700		ZZ2528=ZZ2528+ZZ2528
+007350	003600		ZZ2528=ZZ2528+ZZ2528
+007350	007400		ZZ2528=ZZ2528+ZZ2528
+007350	017000		ZZ2528=ZZ2528+ZZ2528
+007350	004023		0 8192 -ZZ1528
+007351	017000		0 ZZ2528
-	 mark 6146, 57		/70 ophi
+007352	000162		ZZ2529=ZZ2529+ZZ2529
+007352	000344		ZZ2529=ZZ2529+ZZ2529
+007352	000710		ZZ2529=ZZ2529+ZZ2529
+007352	001620		ZZ2529=ZZ2529+ZZ2529
+007352	003440		ZZ2529=ZZ2529+ZZ2529
+007352	007100		ZZ2529=ZZ2529+ZZ2529
+007352	016200		ZZ2529=ZZ2529+ZZ2529
+007352	034400		ZZ2529=ZZ2529+ZZ2529
+007352	003776		0 8192 -ZZ1529
+007353	034400		0 ZZ2529
-	 mark 6158, 198		/71 ophi
+007354	000614		ZZ2530=ZZ2530+ZZ2530
+007354	001430		ZZ2530=ZZ2530+ZZ2530
+007354	003060		ZZ2530=ZZ2530+ZZ2530
+007354	006140		ZZ2530=ZZ2530+ZZ2530
+007354	014300		ZZ2530=ZZ2530+ZZ2530
+007354	030600		ZZ2530=ZZ2530+ZZ2530
+007354	061400		ZZ2530=ZZ2530+ZZ2530
+007354	143000		ZZ2530=ZZ2530+ZZ2530
+007354	003762		0 8192 -ZZ1530
+007355	143000		0 ZZ2530
-	 mark 6170, 473		/102 herc
+007356	001662		ZZ2531=ZZ2531+ZZ2531
+007356	003544		ZZ2531=ZZ2531+ZZ2531
+007356	007310		ZZ2531=ZZ2531+ZZ2531
+007356	016620		ZZ2531=ZZ2531+ZZ2531
+007356	035440		ZZ2531=ZZ2531+ZZ2531
+007356	073100		ZZ2531=ZZ2531+ZZ2531
+007356	166200		ZZ2531=ZZ2531+ZZ2531
+007356	354400		ZZ2531=ZZ2531+ZZ2531
+007356	003746		0 8192 -ZZ1531
+007357	354400		0 ZZ2531
-	 mark 6188, -480	/13 sgtr
+007360	776076		ZZ2532=ZZ2532+ZZ2532
+007360	774174		ZZ2532=ZZ2532+ZZ2532
+007360	770370		ZZ2532=ZZ2532+ZZ2532
+007360	760760		ZZ2532=ZZ2532+ZZ2532
+007360	741740		ZZ2532=ZZ2532+ZZ2532
+007360	703700		ZZ2532=ZZ2532+ZZ2532
+007360	607600		ZZ2532=ZZ2532+ZZ2532
+007360	417400		ZZ2532=ZZ2532+ZZ2532
+007360	003724		0 8192 -ZZ1532
+007361	417400		0 ZZ2532
-	 mark 6234, 76		/74 ophi
+007362	000230		ZZ2533=ZZ2533+ZZ2533
+007362	000460		ZZ2533=ZZ2533+ZZ2533
+007362	001140		ZZ2533=ZZ2533+ZZ2533
+007362	002300		ZZ2533=ZZ2533+ZZ2533
+007362	004600		ZZ2533=ZZ2533+ZZ2533
+007362	011400		ZZ2533=ZZ2533+ZZ2533
+007362	023000		ZZ2533=ZZ2533+ZZ2533
+007362	046000		ZZ2533=ZZ2533+ZZ2533
+007362	003646		0 8192 -ZZ1533
+007363	046000		0 ZZ2533
-	 mark 6235, 499		/106 herc
+007364	001746		ZZ2534=ZZ2534+ZZ2534
+007364	003714		ZZ2534=ZZ2534+ZZ2534
+007364	007630		ZZ2534=ZZ2534+ZZ2534
+007364	017460		ZZ2534=ZZ2534+ZZ2534
+007364	037140		ZZ2534=ZZ2534+ZZ2534
+007364	076300		ZZ2534=ZZ2534+ZZ2534
+007364	174600		ZZ2534=ZZ2534+ZZ2534
+007364	371400		ZZ2534=ZZ2534+ZZ2534
+007364	003645		0 8192 -ZZ1534
+007365	371400		0 ZZ2534
-	 mark 6247, -204	/xi scut
+007366	777146		ZZ2535=ZZ2535+ZZ2535
+007366	776314		ZZ2535=ZZ2535+ZZ2535
+007366	774630		ZZ2535=ZZ2535+ZZ2535
+007366	771460		ZZ2535=ZZ2535+ZZ2535
+007366	763140		ZZ2535=ZZ2535+ZZ2535
+007366	746300		ZZ2535=ZZ2535+ZZ2535
+007366	714600		ZZ2535=ZZ2535+ZZ2535
+007366	631400		ZZ2535=ZZ2535+ZZ2535
+007366	003631		0 8192 -ZZ1535
+007367	631400		0 ZZ2535
-	 mark 6254, -469	/21 sgtr
+007370	776124		ZZ2536=ZZ2536+ZZ2536
+007370	774250		ZZ2536=ZZ2536+ZZ2536
+007370	770520		ZZ2536=ZZ2536+ZZ2536
+007370	761240		ZZ2536=ZZ2536+ZZ2536
+007370	742500		ZZ2536=ZZ2536+ZZ2536
+007370	705200		ZZ2536=ZZ2536+ZZ2536
+007370	612400		ZZ2536=ZZ2536+ZZ2536
+007370	425000		ZZ2536=ZZ2536+ZZ2536
+007370	003622		0 8192 -ZZ1536
+007371	425000		0 ZZ2536
-	 mark 6255, 494		/109 herc
+007372	001734		ZZ2537=ZZ2537+ZZ2537
+007372	003670		ZZ2537=ZZ2537+ZZ2537
+007372	007560		ZZ2537=ZZ2537+ZZ2537
+007372	017340		ZZ2537=ZZ2537+ZZ2537
+007372	036700		ZZ2537=ZZ2537+ZZ2537
+007372	075600		ZZ2537=ZZ2537+ZZ2537
+007372	173400		ZZ2537=ZZ2537+ZZ2537
+007372	367000		ZZ2537=ZZ2537+ZZ2537
+007372	003621		0 8192 -ZZ1537
+007373	367000		0 ZZ2537
-	 mark 6278, -333	/ga scut
+007374	776544		ZZ2538=ZZ2538+ZZ2538
+007374	775310		ZZ2538=ZZ2538+ZZ2538
+007374	772620		ZZ2538=ZZ2538+ZZ2538
+007374	765440		ZZ2538=ZZ2538+ZZ2538
+007374	753100		ZZ2538=ZZ2538+ZZ2538
+007374	726200		ZZ2538=ZZ2538+ZZ2538
+007374	654400		ZZ2538=ZZ2538+ZZ2538
+007374	531000		ZZ2538=ZZ2538+ZZ2538
+007374	003572		0 8192 -ZZ1538
+007375	531000		0 ZZ2538
-	 mark 6313, -189	/al scut
+007376	777204		ZZ2539=ZZ2539+ZZ2539
+007376	776410		ZZ2539=ZZ2539+ZZ2539
+007376	775020		ZZ2539=ZZ2539+ZZ2539
+007376	772040		ZZ2539=ZZ2539+ZZ2539
+007376	764100		ZZ2539=ZZ2539+ZZ2539
+007376	750200		ZZ2539=ZZ2539+ZZ2539
+007376	720400		ZZ2539=ZZ2539+ZZ2539
+007376	641000		ZZ2539=ZZ2539+ZZ2539
+007376	003527		0 8192 -ZZ1539
+007377	641000		0 ZZ2539
-	 mark 6379, 465		/110 herc
+007400	001642		ZZ2540=ZZ2540+ZZ2540
+007400	003504		ZZ2540=ZZ2540+ZZ2540
+007400	007210		ZZ2540=ZZ2540+ZZ2540
+007400	016420		ZZ2540=ZZ2540+ZZ2540
+007400	035040		ZZ2540=ZZ2540+ZZ2540
+007400	072100		ZZ2540=ZZ2540+ZZ2540
+007400	164200		ZZ2540=ZZ2540+ZZ2540
+007400	350400		ZZ2540=ZZ2540+ZZ2540
+007400	003425		0 8192 -ZZ1540
+007401	350400		0 ZZ2540
-	 mark 6382, -110	/be scut
+007402	777442		ZZ2541=ZZ2541+ZZ2541
+007402	777104		ZZ2541=ZZ2541+ZZ2541
+007402	776210		ZZ2541=ZZ2541+ZZ2541
+007402	774420		ZZ2541=ZZ2541+ZZ2541
+007402	771040		ZZ2541=ZZ2541+ZZ2541
+007402	762100		ZZ2541=ZZ2541+ZZ2541
+007402	744200		ZZ2541=ZZ2541+ZZ2541
+007402	710400		ZZ2541=ZZ2541+ZZ2541
+007402	003422		0 8192 -ZZ1541
+007403	710400		0 ZZ2541
-	 mark 6386, 411		/111 herc
+007404	001466		ZZ2542=ZZ2542+ZZ2542
+007404	003154		ZZ2542=ZZ2542+ZZ2542
+007404	006330		ZZ2542=ZZ2542+ZZ2542
+007404	014660		ZZ2542=ZZ2542+ZZ2542
+007404	031540		ZZ2542=ZZ2542+ZZ2542
+007404	063300		ZZ2542=ZZ2542+ZZ2542
+007404	146600		ZZ2542=ZZ2542+ZZ2542
+007404	315400		ZZ2542=ZZ2542+ZZ2542
+007404	003416		0 8192 -ZZ1542
+007405	315400		0 ZZ2542
-	 mark 6436, 93		/63 serp
+007406	000272		ZZ2543=ZZ2543+ZZ2543
+007406	000564		ZZ2543=ZZ2543+ZZ2543
+007406	001350		ZZ2543=ZZ2543+ZZ2543
+007406	002720		ZZ2543=ZZ2543+ZZ2543
+007406	005640		ZZ2543=ZZ2543+ZZ2543
+007406	013500		ZZ2543=ZZ2543+ZZ2543
+007406	027200		ZZ2543=ZZ2543+ZZ2543
+007406	056400		ZZ2543=ZZ2543+ZZ2543
+007406	003334		0 8192 -ZZ1543
+007407	056400		0 ZZ2543
-	 mark 6457, 340		/13 aqil
+007410	001250		ZZ2544=ZZ2544+ZZ2544
+007410	002520		ZZ2544=ZZ2544+ZZ2544
+007410	005240		ZZ2544=ZZ2544+ZZ2544
+007410	012500		ZZ2544=ZZ2544+ZZ2544
+007410	025200		ZZ2544=ZZ2544+ZZ2544
+007410	052400		ZZ2544=ZZ2544+ZZ2544
+007410	125000		ZZ2544=ZZ2544+ZZ2544
+007410	252000		ZZ2544=ZZ2544+ZZ2544
+007410	003307		0 8192 -ZZ1544
+007411	252000		0 ZZ2544
-	 mark 6465, -134	/12 aqil
+007412	777362		ZZ2545=ZZ2545+ZZ2545
+007412	776744		ZZ2545=ZZ2545+ZZ2545
+007412	775710		ZZ2545=ZZ2545+ZZ2545
+007412	773620		ZZ2545=ZZ2545+ZZ2545
+007412	767440		ZZ2545=ZZ2545+ZZ2545
+007412	757100		ZZ2545=ZZ2545+ZZ2545
+007412	736200		ZZ2545=ZZ2545+ZZ2545
+007412	674400		ZZ2545=ZZ2545+ZZ2545
+007412	003277		0 8192 -ZZ1545
+007413	674400		0 ZZ2545
-	 mark 6478, -498	/39 sgtr
+007414	776032		ZZ2546=ZZ2546+ZZ2546
+007414	774064		ZZ2546=ZZ2546+ZZ2546
+007414	770150		ZZ2546=ZZ2546+ZZ2546
+007414	760320		ZZ2546=ZZ2546+ZZ2546
+007414	740640		ZZ2546=ZZ2546+ZZ2546
+007414	701500		ZZ2546=ZZ2546+ZZ2546
+007414	603200		ZZ2546=ZZ2546+ZZ2546
+007414	406400		ZZ2546=ZZ2546+ZZ2546
+007414	003262		0 8192 -ZZ1546
+007415	406400		0 ZZ2546
-	 mark 6553, 483		/ 1 vulp
+007416	001706		ZZ2547=ZZ2547+ZZ2547
+007416	003614		ZZ2547=ZZ2547+ZZ2547
+007416	007430		ZZ2547=ZZ2547+ZZ2547
+007416	017060		ZZ2547=ZZ2547+ZZ2547
+007416	036140		ZZ2547=ZZ2547+ZZ2547
+007416	074300		ZZ2547=ZZ2547+ZZ2547
+007416	170600		ZZ2547=ZZ2547+ZZ2547
+007416	361400		ZZ2547=ZZ2547+ZZ2547
+007416	003147		0 8192 -ZZ1547
+007417	361400		0 ZZ2547
-	 mark 6576, -410	/44 sgtr
+007420	776312		ZZ2548=ZZ2548+ZZ2548
+007420	774624		ZZ2548=ZZ2548+ZZ2548
+007420	771450		ZZ2548=ZZ2548+ZZ2548
+007420	763120		ZZ2548=ZZ2548+ZZ2548
+007420	746240		ZZ2548=ZZ2548+ZZ2548
+007420	714500		ZZ2548=ZZ2548+ZZ2548
+007420	631200		ZZ2548=ZZ2548+ZZ2548
+007420	462400		ZZ2548=ZZ2548+ZZ2548
+007420	003120		0 8192 -ZZ1548
+007421	462400		0 ZZ2548
-	 mark 6576, -368	/46 sgtr
+007422	776436		ZZ2549=ZZ2549+ZZ2549
+007422	775074		ZZ2549=ZZ2549+ZZ2549
+007422	772170		ZZ2549=ZZ2549+ZZ2549
+007422	764360		ZZ2549=ZZ2549+ZZ2549
+007422	750740		ZZ2549=ZZ2549+ZZ2549
+007422	721700		ZZ2549=ZZ2549+ZZ2549
+007422	643600		ZZ2549=ZZ2549+ZZ2549
+007422	507400		ZZ2549=ZZ2549+ZZ2549
+007422	003120		0 8192 -ZZ1549
+007423	507400		0 ZZ2549
-	 mark 6607, 3		/32 aqil
+007424	000006		ZZ2550=ZZ2550+ZZ2550
+007424	000014		ZZ2550=ZZ2550+ZZ2550
+007424	000030		ZZ2550=ZZ2550+ZZ2550
+007424	000060		ZZ2550=ZZ2550+ZZ2550
+007424	000140		ZZ2550=ZZ2550+ZZ2550
+007424	000300		ZZ2550=ZZ2550+ZZ2550
+007424	000600		ZZ2550=ZZ2550+ZZ2550
+007424	001400		ZZ2550=ZZ2550+ZZ2550
+007424	003061		0 8192 -ZZ1550
+007425	001400		0 ZZ2550
-	 mark 6651, 163		/38 aqil
+007426	000506		ZZ2551=ZZ2551+ZZ2551
+007426	001214		ZZ2551=ZZ2551+ZZ2551
+007426	002430		ZZ2551=ZZ2551+ZZ2551
+007426	005060		ZZ2551=ZZ2551+ZZ2551
+007426	012140		ZZ2551=ZZ2551+ZZ2551
+007426	024300		ZZ2551=ZZ2551+ZZ2551
+007426	050600		ZZ2551=ZZ2551+ZZ2551
+007426	121400		ZZ2551=ZZ2551+ZZ2551
+007426	003005		0 8192 -ZZ1551
+007427	121400		0 ZZ2551
-	 mark 6657, 445		/ 9 vulp
+007430	001572		ZZ2552=ZZ2552+ZZ2552
+007430	003364		ZZ2552=ZZ2552+ZZ2552
+007430	006750		ZZ2552=ZZ2552+ZZ2552
+007430	015720		ZZ2552=ZZ2552+ZZ2552
+007430	033640		ZZ2552=ZZ2552+ZZ2552
+007430	067500		ZZ2552=ZZ2552+ZZ2552
+007430	157200		ZZ2552=ZZ2552+ZZ2552
+007430	336400		ZZ2552=ZZ2552+ZZ2552
+007430	002777		0 8192 -ZZ1552
+007431	336400		0 ZZ2552
-	 mark 6665, -35		/41 aqil
+007432	777670		ZZ2553=ZZ2553+ZZ2553
+007432	777560		ZZ2553=ZZ2553+ZZ2553
+007432	777340		ZZ2553=ZZ2553+ZZ2553
+007432	776700		ZZ2553=ZZ2553+ZZ2553
+007432	775600		ZZ2553=ZZ2553+ZZ2553
+007432	773400		ZZ2553=ZZ2553+ZZ2553
+007432	767000		ZZ2553=ZZ2553+ZZ2553
+007432	756000		ZZ2553=ZZ2553+ZZ2553
+007432	002767		0 8192 -ZZ1553
+007433	756000		0 ZZ2553
-	 mark 6688, 405		/ 5 sgte
+007434	001452		ZZ2554=ZZ2554+ZZ2554
+007434	003124		ZZ2554=ZZ2554+ZZ2554
+007434	006250		ZZ2554=ZZ2554+ZZ2554
+007434	014520		ZZ2554=ZZ2554+ZZ2554
+007434	031240		ZZ2554=ZZ2554+ZZ2554
+007434	062500		ZZ2554=ZZ2554+ZZ2554
+007434	145200		ZZ2554=ZZ2554+ZZ2554
+007434	312400		ZZ2554=ZZ2554+ZZ2554
+007434	002740		0 8192 -ZZ1554
+007435	312400		0 ZZ2554
-	 mark 6693, 393		/ 6 sgte
+007436	001422		ZZ2555=ZZ2555+ZZ2555
+007436	003044		ZZ2555=ZZ2555+ZZ2555
+007436	006110		ZZ2555=ZZ2555+ZZ2555
+007436	014220		ZZ2555=ZZ2555+ZZ2555
+007436	030440		ZZ2555=ZZ2555+ZZ2555
+007436	061100		ZZ2555=ZZ2555+ZZ2555
+007436	142200		ZZ2555=ZZ2555+ZZ2555
+007436	304400		ZZ2555=ZZ2555+ZZ2555
+007436	002733		0 8192 -ZZ1555
+007437	304400		0 ZZ2555
-	 mark 6730, 416		/ 7 sgte
+007440	001500		ZZ2556=ZZ2556+ZZ2556
+007440	003200		ZZ2556=ZZ2556+ZZ2556
+007440	006400		ZZ2556=ZZ2556+ZZ2556
+007440	015000		ZZ2556=ZZ2556+ZZ2556
+007440	032000		ZZ2556=ZZ2556+ZZ2556
+007440	064000		ZZ2556=ZZ2556+ZZ2556
+007440	150000		ZZ2556=ZZ2556+ZZ2556
+007440	320000		ZZ2556=ZZ2556+ZZ2556
+007440	002666		0 8192 -ZZ1556
+007441	320000		0 ZZ2556
-	 mark 6739, 430		/ 8 sgte
+007442	001534		ZZ2557=ZZ2557+ZZ2557
+007442	003270		ZZ2557=ZZ2557+ZZ2557
+007442	006560		ZZ2557=ZZ2557+ZZ2557
+007442	015340		ZZ2557=ZZ2557+ZZ2557
+007442	032700		ZZ2557=ZZ2557+ZZ2557
+007442	065600		ZZ2557=ZZ2557+ZZ2557
+007442	153400		ZZ2557=ZZ2557+ZZ2557
+007442	327000		ZZ2557=ZZ2557+ZZ2557
+007442	002655		0 8192 -ZZ1557
+007443	327000		0 ZZ2557
-	 mark 6755, 17		/55 aqil
+007444	000042		ZZ2558=ZZ2558+ZZ2558
+007444	000104		ZZ2558=ZZ2558+ZZ2558
+007444	000210		ZZ2558=ZZ2558+ZZ2558
+007444	000420		ZZ2558=ZZ2558+ZZ2558
+007444	001040		ZZ2558=ZZ2558+ZZ2558
+007444	002100		ZZ2558=ZZ2558+ZZ2558
+007444	004200		ZZ2558=ZZ2558+ZZ2558
+007444	010400		ZZ2558=ZZ2558+ZZ2558
+007444	002635		0 8192 -ZZ1558
+007445	010400		0 ZZ2558
-	 mark 6766, 187		/59 aqil
+007446	000566		ZZ2559=ZZ2559+ZZ2559
+007446	001354		ZZ2559=ZZ2559+ZZ2559
+007446	002730		ZZ2559=ZZ2559+ZZ2559
+007446	005660		ZZ2559=ZZ2559+ZZ2559
+007446	013540		ZZ2559=ZZ2559+ZZ2559
+007446	027300		ZZ2559=ZZ2559+ZZ2559
+007446	056600		ZZ2559=ZZ2559+ZZ2559
+007446	135400		ZZ2559=ZZ2559+ZZ2559
+007446	002622		0 8192 -ZZ1559
+007447	135400		0 ZZ2559
-	 mark 6772, 140		/60 aqil
+007450	000430		ZZ2560=ZZ2560+ZZ2560
+007450	001060		ZZ2560=ZZ2560+ZZ2560
+007450	002140		ZZ2560=ZZ2560+ZZ2560
+007450	004300		ZZ2560=ZZ2560+ZZ2560
+007450	010600		ZZ2560=ZZ2560+ZZ2560
+007450	021400		ZZ2560=ZZ2560+ZZ2560
+007450	043000		ZZ2560=ZZ2560+ZZ2560
+007450	106000		ZZ2560=ZZ2560+ZZ2560
+007450	002614		0 8192 -ZZ1560
+007451	106000		0 ZZ2560
-	 mark 6882, 339		/67 aqil
+007452	001246		ZZ2561=ZZ2561+ZZ2561
+007452	002514		ZZ2561=ZZ2561+ZZ2561
+007452	005230		ZZ2561=ZZ2561+ZZ2561
+007452	012460		ZZ2561=ZZ2561+ZZ2561
+007452	025140		ZZ2561=ZZ2561+ZZ2561
+007452	052300		ZZ2561=ZZ2561+ZZ2561
+007452	124600		ZZ2561=ZZ2561+ZZ2561
+007452	251400		ZZ2561=ZZ2561+ZZ2561
+007452	002436		0 8192 -ZZ1561
+007453	251400		0 ZZ2561
-	 mark 6896, -292	/ 5 capr
+007454	776666		ZZ2562=ZZ2562+ZZ2562
+007454	775554		ZZ2562=ZZ2562+ZZ2562
+007454	773330		ZZ2562=ZZ2562+ZZ2562
+007454	766660		ZZ2562=ZZ2562+ZZ2562
+007454	755540		ZZ2562=ZZ2562+ZZ2562
+007454	733300		ZZ2562=ZZ2562+ZZ2562
+007454	666600		ZZ2562=ZZ2562+ZZ2562
+007454	555400		ZZ2562=ZZ2562+ZZ2562
+007454	002420		0 8192 -ZZ1562
+007455	555400		0 ZZ2562
-	 mark 6898, -292	/ 6 capr
+007456	776666		ZZ2563=ZZ2563+ZZ2563
+007456	775554		ZZ2563=ZZ2563+ZZ2563
+007456	773330		ZZ2563=ZZ2563+ZZ2563
+007456	766660		ZZ2563=ZZ2563+ZZ2563
+007456	755540		ZZ2563=ZZ2563+ZZ2563
+007456	733300		ZZ2563=ZZ2563+ZZ2563
+007456	666600		ZZ2563=ZZ2563+ZZ2563
+007456	555400		ZZ2563=ZZ2563+ZZ2563
+007456	002416		0 8192 -ZZ1563
+007457	555400		0 ZZ2563
-	 mark 6913, -297	/ 8 capr
+007460	776654		ZZ2564=ZZ2564+ZZ2564
+007460	775530		ZZ2564=ZZ2564+ZZ2564
+007460	773260		ZZ2564=ZZ2564+ZZ2564
+007460	766540		ZZ2564=ZZ2564+ZZ2564
+007460	755300		ZZ2564=ZZ2564+ZZ2564
+007460	732600		ZZ2564=ZZ2564+ZZ2564
+007460	665400		ZZ2564=ZZ2564+ZZ2564
+007460	553000		ZZ2564=ZZ2564+ZZ2564
+007460	002377		0 8192 -ZZ1564
+007461	553000		0 ZZ2564
-	 mark 6958, -413	/11 capr
+007462	776304		ZZ2565=ZZ2565+ZZ2565
+007462	774610		ZZ2565=ZZ2565+ZZ2565
+007462	771420		ZZ2565=ZZ2565+ZZ2565
+007462	763040		ZZ2565=ZZ2565+ZZ2565
+007462	746100		ZZ2565=ZZ2565+ZZ2565
+007462	714200		ZZ2565=ZZ2565+ZZ2565
+007462	630400		ZZ2565=ZZ2565+ZZ2565
+007462	461000		ZZ2565=ZZ2565+ZZ2565
+007462	002322		0 8192 -ZZ1565
+007463	461000		0 ZZ2565
-	 mark 6988, 250		/ 2 dlph
+007464	000764		ZZ2566=ZZ2566+ZZ2566
+007464	001750		ZZ2566=ZZ2566+ZZ2566
+007464	003720		ZZ2566=ZZ2566+ZZ2566
+007464	007640		ZZ2566=ZZ2566+ZZ2566
+007464	017500		ZZ2566=ZZ2566+ZZ2566
+007464	037200		ZZ2566=ZZ2566+ZZ2566
+007464	076400		ZZ2566=ZZ2566+ZZ2566
+007464	175000		ZZ2566=ZZ2566+ZZ2566
+007464	002264		0 8192 -ZZ1566
+007465	175000		0 ZZ2566
-	 mark 7001, 326		/ 4 dlph
+007466	001214		ZZ2567=ZZ2567+ZZ2567
+007466	002430		ZZ2567=ZZ2567+ZZ2567
+007466	005060		ZZ2567=ZZ2567+ZZ2567
+007466	012140		ZZ2567=ZZ2567+ZZ2567
+007466	024300		ZZ2567=ZZ2567+ZZ2567
+007466	050600		ZZ2567=ZZ2567+ZZ2567
+007466	121400		ZZ2567=ZZ2567+ZZ2567
+007466	243000		ZZ2567=ZZ2567+ZZ2567
+007466	002247		0 8192 -ZZ1567
+007467	243000		0 ZZ2567
-	 mark 7015, -33		/71 aqil
+007470	777674		ZZ2568=ZZ2568+ZZ2568
+007470	777570		ZZ2568=ZZ2568+ZZ2568
+007470	777360		ZZ2568=ZZ2568+ZZ2568
+007470	776740		ZZ2568=ZZ2568+ZZ2568
+007470	775700		ZZ2568=ZZ2568+ZZ2568
+007470	773600		ZZ2568=ZZ2568+ZZ2568
+007470	767400		ZZ2568=ZZ2568+ZZ2568
+007470	757000		ZZ2568=ZZ2568+ZZ2568
+007470	002231		0 8192 -ZZ1568
+007471	757000		0 ZZ2568
-	 mark 7020, 475		/29 vulp
+007472	001666		ZZ2569=ZZ2569+ZZ2569
+007472	003554		ZZ2569=ZZ2569+ZZ2569
+007472	007330		ZZ2569=ZZ2569+ZZ2569
+007472	016660		ZZ2569=ZZ2569+ZZ2569
+007472	035540		ZZ2569=ZZ2569+ZZ2569
+007472	073300		ZZ2569=ZZ2569+ZZ2569
+007472	166600		ZZ2569=ZZ2569+ZZ2569
+007472	355400		ZZ2569=ZZ2569+ZZ2569
+007472	002224		0 8192 -ZZ1569
+007473	355400		0 ZZ2569
-	 mark 7026, 354		/ 9 dlph
+007474	001304		ZZ2570=ZZ2570+ZZ2570
+007474	002610		ZZ2570=ZZ2570+ZZ2570
+007474	005420		ZZ2570=ZZ2570+ZZ2570
+007474	013040		ZZ2570=ZZ2570+ZZ2570
+007474	026100		ZZ2570=ZZ2570+ZZ2570
+007474	054200		ZZ2570=ZZ2570+ZZ2570
+007474	130400		ZZ2570=ZZ2570+ZZ2570
+007474	261000		ZZ2570=ZZ2570+ZZ2570
+007474	002216		0 8192 -ZZ1570
+007475	261000		0 ZZ2570
-	 mark 7047, 335		/11 dlph
+007476	001236		ZZ2571=ZZ2571+ZZ2571
+007476	002474		ZZ2571=ZZ2571+ZZ2571
+007476	005170		ZZ2571=ZZ2571+ZZ2571
+007476	012360		ZZ2571=ZZ2571+ZZ2571
+007476	024740		ZZ2571=ZZ2571+ZZ2571
+007476	051700		ZZ2571=ZZ2571+ZZ2571
+007476	123600		ZZ2571=ZZ2571+ZZ2571
+007476	247400		ZZ2571=ZZ2571+ZZ2571
+007476	002171		0 8192 -ZZ1571
+007477	247400		0 ZZ2571
-	 mark 7066, 359		/12 dlph
+007500	001316		ZZ2572=ZZ2572+ZZ2572
+007500	002634		ZZ2572=ZZ2572+ZZ2572
+007500	005470		ZZ2572=ZZ2572+ZZ2572
+007500	013160		ZZ2572=ZZ2572+ZZ2572
+007500	026340		ZZ2572=ZZ2572+ZZ2572
+007500	054700		ZZ2572=ZZ2572+ZZ2572
+007500	131600		ZZ2572=ZZ2572+ZZ2572
+007500	263400		ZZ2572=ZZ2572+ZZ2572
+007500	002146		0 8192 -ZZ1572
+007501	263400		0 ZZ2572
-	 mark 7067, -225	/ 2 aqar
+007502	777074		ZZ2573=ZZ2573+ZZ2573
+007502	776170		ZZ2573=ZZ2573+ZZ2573
+007502	774360		ZZ2573=ZZ2573+ZZ2573
+007502	770740		ZZ2573=ZZ2573+ZZ2573
+007502	761700		ZZ2573=ZZ2573+ZZ2573
+007502	743600		ZZ2573=ZZ2573+ZZ2573
+007502	707400		ZZ2573=ZZ2573+ZZ2573
+007502	617000		ZZ2573=ZZ2573+ZZ2573
+007502	002145		0 8192 -ZZ1573
+007503	617000		0 ZZ2573
-	 mark 7068, -123	/ 3 aqar
+007504	777410		ZZ2574=ZZ2574+ZZ2574
+007504	777020		ZZ2574=ZZ2574+ZZ2574
+007504	776040		ZZ2574=ZZ2574+ZZ2574
+007504	774100		ZZ2574=ZZ2574+ZZ2574
+007504	770200		ZZ2574=ZZ2574+ZZ2574
+007504	760400		ZZ2574=ZZ2574+ZZ2574
+007504	741000		ZZ2574=ZZ2574+ZZ2574
+007504	702000		ZZ2574=ZZ2574+ZZ2574
+007504	002144		0 8192 -ZZ1574
+007505	702000		0 ZZ2574
-	 mark 7096, -213	/ 6 aqar
+007506	777124		ZZ2575=ZZ2575+ZZ2575
+007506	776250		ZZ2575=ZZ2575+ZZ2575
+007506	774520		ZZ2575=ZZ2575+ZZ2575
+007506	771240		ZZ2575=ZZ2575+ZZ2575
+007506	762500		ZZ2575=ZZ2575+ZZ2575
+007506	745200		ZZ2575=ZZ2575+ZZ2575
+007506	712400		ZZ2575=ZZ2575+ZZ2575
+007506	625000		ZZ2575=ZZ2575+ZZ2575
+007506	002110		0 8192 -ZZ1575
+007507	625000		0 ZZ2575
-	 mark 7161, -461	/22 capr
+007510	776144		ZZ2576=ZZ2576+ZZ2576
+007510	774310		ZZ2576=ZZ2576+ZZ2576
+007510	770620		ZZ2576=ZZ2576+ZZ2576
+007510	761440		ZZ2576=ZZ2576+ZZ2576
+007510	743100		ZZ2576=ZZ2576+ZZ2576
+007510	706200		ZZ2576=ZZ2576+ZZ2576
+007510	614400		ZZ2576=ZZ2576+ZZ2576
+007510	431000		ZZ2576=ZZ2576+ZZ2576
+007510	002007		0 8192 -ZZ1576
+007511	431000		0 ZZ2576
-	 mark 7170, -401	/23 capr
+007512	776334		ZZ2577=ZZ2577+ZZ2577
+007512	774670		ZZ2577=ZZ2577+ZZ2577
+007512	771560		ZZ2577=ZZ2577+ZZ2577
+007512	763340		ZZ2577=ZZ2577+ZZ2577
+007512	746700		ZZ2577=ZZ2577+ZZ2577
+007512	715600		ZZ2577=ZZ2577+ZZ2577
+007512	633400		ZZ2577=ZZ2577+ZZ2577
+007512	467000		ZZ2577=ZZ2577+ZZ2577
+007512	001776		0 8192 -ZZ1577
+007513	467000		0 ZZ2577
-	 mark 7192, -268	/13 capr
+007514	776746		ZZ2578=ZZ2578+ZZ2578
+007514	775714		ZZ2578=ZZ2578+ZZ2578
+007514	773630		ZZ2578=ZZ2578+ZZ2578
+007514	767460		ZZ2578=ZZ2578+ZZ2578
+007514	757140		ZZ2578=ZZ2578+ZZ2578
+007514	736300		ZZ2578=ZZ2578+ZZ2578
+007514	674600		ZZ2578=ZZ2578+ZZ2578
+007514	571400		ZZ2578=ZZ2578+ZZ2578
+007514	001750		0 8192 -ZZ1578
+007515	571400		0 ZZ2578
-	 mark 7199, 222		/ 5 equl
+007516	000674		ZZ2579=ZZ2579+ZZ2579
+007516	001570		ZZ2579=ZZ2579+ZZ2579
+007516	003360		ZZ2579=ZZ2579+ZZ2579
+007516	006740		ZZ2579=ZZ2579+ZZ2579
+007516	015700		ZZ2579=ZZ2579+ZZ2579
+007516	033600		ZZ2579=ZZ2579+ZZ2579
+007516	067400		ZZ2579=ZZ2579+ZZ2579
+007516	157000		ZZ2579=ZZ2579+ZZ2579
+007516	001741		0 8192 -ZZ1579
+007517	157000		0 ZZ2579
-	 mark 7223, 219		/ 7 equl
+007520	000666		ZZ2580=ZZ2580+ZZ2580
+007520	001554		ZZ2580=ZZ2580+ZZ2580
+007520	003330		ZZ2580=ZZ2580+ZZ2580
+007520	006660		ZZ2580=ZZ2580+ZZ2580
+007520	015540		ZZ2580=ZZ2580+ZZ2580
+007520	033300		ZZ2580=ZZ2580+ZZ2580
+007520	066600		ZZ2580=ZZ2580+ZZ2580
+007520	155400		ZZ2580=ZZ2580+ZZ2580
+007520	001711		0 8192 -ZZ1580
+007521	155400		0 ZZ2580
-	 mark 7230, 110		/ 8 equl
+007522	000334		ZZ2581=ZZ2581+ZZ2581
+007522	000670		ZZ2581=ZZ2581+ZZ2581
+007522	001560		ZZ2581=ZZ2581+ZZ2581
+007522	003340		ZZ2581=ZZ2581+ZZ2581
+007522	006700		ZZ2581=ZZ2581+ZZ2581
+007522	015600		ZZ2581=ZZ2581+ZZ2581
+007522	033400		ZZ2581=ZZ2581+ZZ2581
+007522	067000		ZZ2581=ZZ2581+ZZ2581
+007522	001702		0 8192 -ZZ1581
+007523	067000		0 ZZ2581
-	 mark 7263, -393	/32 capr
+007524	776354		ZZ2582=ZZ2582+ZZ2582
+007524	774730		ZZ2582=ZZ2582+ZZ2582
+007524	771660		ZZ2582=ZZ2582+ZZ2582
+007524	763540		ZZ2582=ZZ2582+ZZ2582
+007524	747300		ZZ2582=ZZ2582+ZZ2582
+007524	716600		ZZ2582=ZZ2582+ZZ2582
+007524	635400		ZZ2582=ZZ2582+ZZ2582
+007524	473000		ZZ2582=ZZ2582+ZZ2582
+007524	001641		0 8192 -ZZ1582
+007525	473000		0 ZZ2582
-	 mark 7267, 441		/ 1 pegs
+007526	001562		ZZ2583=ZZ2583+ZZ2583
+007526	003344		ZZ2583=ZZ2583+ZZ2583
+007526	006710		ZZ2583=ZZ2583+ZZ2583
+007526	015620		ZZ2583=ZZ2583+ZZ2583
+007526	033440		ZZ2583=ZZ2583+ZZ2583
+007526	067100		ZZ2583=ZZ2583+ZZ2583
+007526	156200		ZZ2583=ZZ2583+ZZ2583
+007526	334400		ZZ2583=ZZ2583+ZZ2583
+007526	001635		0 8192 -ZZ1583
+007527	334400		0 ZZ2583
-	 mark 7299, -506	/36 capr
+007530	776012		ZZ2584=ZZ2584+ZZ2584
+007530	774024		ZZ2584=ZZ2584+ZZ2584
+007530	770050		ZZ2584=ZZ2584+ZZ2584
+007530	760120		ZZ2584=ZZ2584+ZZ2584
+007530	740240		ZZ2584=ZZ2584+ZZ2584
+007530	700500		ZZ2584=ZZ2584+ZZ2584
+007530	601200		ZZ2584=ZZ2584+ZZ2584
+007530	402400		ZZ2584=ZZ2584+ZZ2584
+007530	001575		0 8192 -ZZ1584
+007531	402400		0 ZZ2584
-	 mark 7347, -453	/39 capr
+007532	776164		ZZ2585=ZZ2585+ZZ2585
+007532	774350		ZZ2585=ZZ2585+ZZ2585
+007532	770720		ZZ2585=ZZ2585+ZZ2585
+007532	761640		ZZ2585=ZZ2585+ZZ2585
+007532	743500		ZZ2585=ZZ2585+ZZ2585
+007532	707200		ZZ2585=ZZ2585+ZZ2585
+007532	616400		ZZ2585=ZZ2585+ZZ2585
+007532	435000		ZZ2585=ZZ2585+ZZ2585
+007532	001515		0 8192 -ZZ1585
+007533	435000		0 ZZ2585
-	 mark 7353, -189	/23 aqar
+007534	777204		ZZ2586=ZZ2586+ZZ2586
+007534	776410		ZZ2586=ZZ2586+ZZ2586
+007534	775020		ZZ2586=ZZ2586+ZZ2586
+007534	772040		ZZ2586=ZZ2586+ZZ2586
+007534	764100		ZZ2586=ZZ2586+ZZ2586
+007534	750200		ZZ2586=ZZ2586+ZZ2586
+007534	720400		ZZ2586=ZZ2586+ZZ2586
+007534	641000		ZZ2586=ZZ2586+ZZ2586
+007534	001507		0 8192 -ZZ1586
+007535	641000		0 ZZ2586
-	 mark 7365, -390	/40 capr
+007536	776362		ZZ2587=ZZ2587+ZZ2587
+007536	774744		ZZ2587=ZZ2587+ZZ2587
+007536	771710		ZZ2587=ZZ2587+ZZ2587
+007536	763620		ZZ2587=ZZ2587+ZZ2587
+007536	747440		ZZ2587=ZZ2587+ZZ2587
+007536	717100		ZZ2587=ZZ2587+ZZ2587
+007536	636200		ZZ2587=ZZ2587+ZZ2587
+007536	474400		ZZ2587=ZZ2587+ZZ2587
+007536	001473		0 8192 -ZZ1587
+007537	474400		0 ZZ2587
-	 mark 7379, -440	/43 capr
+007540	776216		ZZ2588=ZZ2588+ZZ2588
+007540	774434		ZZ2588=ZZ2588+ZZ2588
+007540	771070		ZZ2588=ZZ2588+ZZ2588
+007540	762160		ZZ2588=ZZ2588+ZZ2588
+007540	744340		ZZ2588=ZZ2588+ZZ2588
+007540	710700		ZZ2588=ZZ2588+ZZ2588
+007540	621600		ZZ2588=ZZ2588+ZZ2588
+007540	443400		ZZ2588=ZZ2588+ZZ2588
+007540	001455		0 8192 -ZZ1588
+007541	443400		0 ZZ2588
-	 mark 7394, 384		/ 9 pegs
+007542	001400		ZZ2589=ZZ2589+ZZ2589
+007542	003000		ZZ2589=ZZ2589+ZZ2589
+007542	006000		ZZ2589=ZZ2589+ZZ2589
+007542	014000		ZZ2589=ZZ2589+ZZ2589
+007542	030000		ZZ2589=ZZ2589+ZZ2589
+007542	060000		ZZ2589=ZZ2589+ZZ2589
+007542	140000		ZZ2589=ZZ2589+ZZ2589
+007542	300000		ZZ2589=ZZ2589+ZZ2589
+007542	001436		0 8192 -ZZ1589
+007543	300000		0 ZZ2589
-	 mark 7499, -60		/31 aquar
+007544	777606		ZZ2590=ZZ2590+ZZ2590
+007544	777414		ZZ2590=ZZ2590+ZZ2590
+007544	777030		ZZ2590=ZZ2590+ZZ2590
+007544	776060		ZZ2590=ZZ2590+ZZ2590
+007544	774140		ZZ2590=ZZ2590+ZZ2590
+007544	770300		ZZ2590=ZZ2590+ZZ2590
+007544	760600		ZZ2590=ZZ2590+ZZ2590
+007544	741400		ZZ2590=ZZ2590+ZZ2590
+007544	001265		0 8192 -ZZ1590
+007545	741400		0 ZZ2590
-	 mark 7513, 104		/22 pegs
+007546	000320		ZZ2591=ZZ2591+ZZ2591
+007546	000640		ZZ2591=ZZ2591+ZZ2591
+007546	001500		ZZ2591=ZZ2591+ZZ2591
+007546	003200		ZZ2591=ZZ2591+ZZ2591
+007546	006400		ZZ2591=ZZ2591+ZZ2591
+007546	015000		ZZ2591=ZZ2591+ZZ2591
+007546	032000		ZZ2591=ZZ2591+ZZ2591
+007546	064000		ZZ2591=ZZ2591+ZZ2591
+007546	001247		0 8192 -ZZ1591
+007547	064000		0 ZZ2591
-	 mark 7515, -327	/33 aqar
+007550	776560		ZZ2592=ZZ2592+ZZ2592
+007550	775340		ZZ2592=ZZ2592+ZZ2592
+007550	772700		ZZ2592=ZZ2592+ZZ2592
+007550	765600		ZZ2592=ZZ2592+ZZ2592
+007550	753400		ZZ2592=ZZ2592+ZZ2592
+007550	727000		ZZ2592=ZZ2592+ZZ2592
+007550	656000		ZZ2592=ZZ2592+ZZ2592
+007550	534000		ZZ2592=ZZ2592+ZZ2592
+007550	001245		0 8192 -ZZ1592
+007551	534000		0 ZZ2592
-	 mark 7575, -189	/43 aqar
+007552	777204		ZZ2593=ZZ2593+ZZ2593
+007552	776410		ZZ2593=ZZ2593+ZZ2593
+007552	775020		ZZ2593=ZZ2593+ZZ2593
+007552	772040		ZZ2593=ZZ2593+ZZ2593
+007552	764100		ZZ2593=ZZ2593+ZZ2593
+007552	750200		ZZ2593=ZZ2593+ZZ2593
+007552	720400		ZZ2593=ZZ2593+ZZ2593
+007552	641000		ZZ2593=ZZ2593+ZZ2593
+007552	001151		0 8192 -ZZ1593
+007553	641000		0 ZZ2593
-	 mark 7603, -43		/48 aqar
+007554	777650		ZZ2594=ZZ2594+ZZ2594
+007554	777520		ZZ2594=ZZ2594+ZZ2594
+007554	777240		ZZ2594=ZZ2594+ZZ2594
+007554	776500		ZZ2594=ZZ2594+ZZ2594
+007554	775200		ZZ2594=ZZ2594+ZZ2594
+007554	772400		ZZ2594=ZZ2594+ZZ2594
+007554	765000		ZZ2594=ZZ2594+ZZ2594
+007554	752000		ZZ2594=ZZ2594+ZZ2594
+007554	001115		0 8192 -ZZ1594
+007555	752000		0 ZZ2594
-	 mark 7604, 266		/31 pegs
+007556	001024		ZZ2595=ZZ2595+ZZ2595
+007556	002050		ZZ2595=ZZ2595+ZZ2595
+007556	004120		ZZ2595=ZZ2595+ZZ2595
+007556	010240		ZZ2595=ZZ2595+ZZ2595
+007556	020500		ZZ2595=ZZ2595+ZZ2595
+007556	041200		ZZ2595=ZZ2595+ZZ2595
+007556	102400		ZZ2595=ZZ2595+ZZ2595
+007556	205000		ZZ2595=ZZ2595+ZZ2595
+007556	001114		0 8192 -ZZ1595
+007557	205000		0 ZZ2595
-	 mark 7624, 20		/52 aquar
+007560	000050		ZZ2596=ZZ2596+ZZ2596
+007560	000120		ZZ2596=ZZ2596+ZZ2596
+007560	000240		ZZ2596=ZZ2596+ZZ2596
+007560	000500		ZZ2596=ZZ2596+ZZ2596
+007560	001200		ZZ2596=ZZ2596+ZZ2596
+007560	002400		ZZ2596=ZZ2596+ZZ2596
+007560	005000		ZZ2596=ZZ2596+ZZ2596
+007560	012000		ZZ2596=ZZ2596+ZZ2596
+007560	001070		0 8192 -ZZ1596
+007561	012000		0 ZZ2596
-	 mark 7639, 96		/35 pegs
+007562	000300		ZZ2597=ZZ2597+ZZ2597
+007562	000600		ZZ2597=ZZ2597+ZZ2597
+007562	001400		ZZ2597=ZZ2597+ZZ2597
+007562	003000		ZZ2597=ZZ2597+ZZ2597
+007562	006000		ZZ2597=ZZ2597+ZZ2597
+007562	014000		ZZ2597=ZZ2597+ZZ2597
+007562	030000		ZZ2597=ZZ2597+ZZ2597
+007562	060000		ZZ2597=ZZ2597+ZZ2597
+007562	001051		0 8192 -ZZ1597
+007563	060000		0 ZZ2597
-	 mark 7654, -255	/57 aqar
+007564	777000		ZZ2598=ZZ2598+ZZ2598
+007564	776000		ZZ2598=ZZ2598+ZZ2598
+007564	774000		ZZ2598=ZZ2598+ZZ2598
+007564	770000		ZZ2598=ZZ2598+ZZ2598
+007564	760000		ZZ2598=ZZ2598+ZZ2598
+007564	740000		ZZ2598=ZZ2598+ZZ2598
+007564	700000		ZZ2598=ZZ2598+ZZ2598
+007564	600000		ZZ2598=ZZ2598+ZZ2598
+007564	001032		0 8192 -ZZ1598
+007565	600000		0 ZZ2598
-	 mark 7681, -14		/62 aqar
+007566	777742		ZZ2599=ZZ2599+ZZ2599
+007566	777704		ZZ2599=ZZ2599+ZZ2599
+007566	777610		ZZ2599=ZZ2599+ZZ2599
+007566	777420		ZZ2599=ZZ2599+ZZ2599
+007566	777040		ZZ2599=ZZ2599+ZZ2599
+007566	776100		ZZ2599=ZZ2599+ZZ2599
+007566	774200		ZZ2599=ZZ2599+ZZ2599
+007566	770400		ZZ2599=ZZ2599+ZZ2599
+007566	000777		0 8192 -ZZ1599
+007567	770400		0 ZZ2599
-	 mark 7727, -440	/66 aqar
+007570	776216		ZZ2600=ZZ2600+ZZ2600
+007570	774434		ZZ2600=ZZ2600+ZZ2600
+007570	771070		ZZ2600=ZZ2600+ZZ2600
+007570	762160		ZZ2600=ZZ2600+ZZ2600
+007570	744340		ZZ2600=ZZ2600+ZZ2600
+007570	710700		ZZ2600=ZZ2600+ZZ2600
+007570	621600		ZZ2600=ZZ2600+ZZ2600
+007570	443400		ZZ2600=ZZ2600+ZZ2600
+007570	000721		0 8192 -ZZ1600
+007571	443400		0 ZZ2600
-	 mark 7747, 266		/46 pegs
+007572	001024		ZZ2601=ZZ2601+ZZ2601
+007572	002050		ZZ2601=ZZ2601+ZZ2601
+007572	004120		ZZ2601=ZZ2601+ZZ2601
+007572	010240		ZZ2601=ZZ2601+ZZ2601
+007572	020500		ZZ2601=ZZ2601+ZZ2601
+007572	041200		ZZ2601=ZZ2601+ZZ2601
+007572	102400		ZZ2601=ZZ2601+ZZ2601
+007572	205000		ZZ2601=ZZ2601+ZZ2601
+007572	000675		0 8192 -ZZ1601
+007573	205000		0 ZZ2601
-	 mark 7761, -321	/71 aqar
+007574	776574		ZZ2602=ZZ2602+ZZ2602
+007574	775370		ZZ2602=ZZ2602+ZZ2602
+007574	772760		ZZ2602=ZZ2602+ZZ2602
+007574	765740		ZZ2602=ZZ2602+ZZ2602
+007574	753700		ZZ2602=ZZ2602+ZZ2602
+007574	727600		ZZ2602=ZZ2602+ZZ2602
+007574	657400		ZZ2602=ZZ2602+ZZ2602
+007574	537000		ZZ2602=ZZ2602+ZZ2602
+007574	000657		0 8192 -ZZ1602
+007575	537000		0 ZZ2602
-	 mark 7779, -185	/73 aqar
+007576	777214		ZZ2603=ZZ2603+ZZ2603
+007576	776430		ZZ2603=ZZ2603+ZZ2603
+007576	775060		ZZ2603=ZZ2603+ZZ2603
+007576	772140		ZZ2603=ZZ2603+ZZ2603
+007576	764300		ZZ2603=ZZ2603+ZZ2603
+007576	750600		ZZ2603=ZZ2603+ZZ2603
+007576	721400		ZZ2603=ZZ2603+ZZ2603
+007576	643000		ZZ2603=ZZ2603+ZZ2603
+007576	000635		0 8192 -ZZ1603
+007577	643000		0 ZZ2603
-	 mark 7795, 189		/50 pegs
+007600	000572		ZZ2604=ZZ2604+ZZ2604
+007600	001364		ZZ2604=ZZ2604+ZZ2604
+007600	002750		ZZ2604=ZZ2604+ZZ2604
+007600	005720		ZZ2604=ZZ2604+ZZ2604
+007600	013640		ZZ2604=ZZ2604+ZZ2604
+007600	027500		ZZ2604=ZZ2604+ZZ2604
+007600	057200		ZZ2604=ZZ2604+ZZ2604
+007600	136400		ZZ2604=ZZ2604+ZZ2604
+007600	000615		0 8192 -ZZ1604
+007601	136400		0 ZZ2604
-	 mark 7844, 75		/ 4 pisc
+007602	000226		ZZ2605=ZZ2605+ZZ2605
+007602	000454		ZZ2605=ZZ2605+ZZ2605
+007602	001130		ZZ2605=ZZ2605+ZZ2605
+007602	002260		ZZ2605=ZZ2605+ZZ2605
+007602	004540		ZZ2605=ZZ2605+ZZ2605
+007602	011300		ZZ2605=ZZ2605+ZZ2605
+007602	022600		ZZ2605=ZZ2605+ZZ2605
+007602	045400		ZZ2605=ZZ2605+ZZ2605
+007602	000534		0 8192 -ZZ1605
+007603	045400		0 ZZ2605
-	 mark 7862, 202		/55 pegs
+007604	000624		ZZ2606=ZZ2606+ZZ2606
+007604	001450		ZZ2606=ZZ2606+ZZ2606
+007604	003120		ZZ2606=ZZ2606+ZZ2606
+007604	006240		ZZ2606=ZZ2606+ZZ2606
+007604	014500		ZZ2606=ZZ2606+ZZ2606
+007604	031200		ZZ2606=ZZ2606+ZZ2606
+007604	062400		ZZ2606=ZZ2606+ZZ2606
+007604	145000		ZZ2606=ZZ2606+ZZ2606
+007604	000512		0 8192 -ZZ1606
+007605	145000		0 ZZ2606
-	 mark 7874, -494	/88 aqar
+007606	776042		ZZ2607=ZZ2607+ZZ2607
+007606	774104		ZZ2607=ZZ2607+ZZ2607
+007606	770210		ZZ2607=ZZ2607+ZZ2607
+007606	760420		ZZ2607=ZZ2607+ZZ2607
+007606	741040		ZZ2607=ZZ2607+ZZ2607
+007606	702100		ZZ2607=ZZ2607+ZZ2607
+007606	604200		ZZ2607=ZZ2607+ZZ2607
+007606	410400		ZZ2607=ZZ2607+ZZ2607
+007606	000476		0 8192 -ZZ1607
+007607	410400		0 ZZ2607
-	 mark 7903, -150	/90 aqar
+007610	777322		ZZ2608=ZZ2608+ZZ2608
+007610	776644		ZZ2608=ZZ2608+ZZ2608
+007610	775510		ZZ2608=ZZ2608+ZZ2608
+007610	773220		ZZ2608=ZZ2608+ZZ2608
+007610	766440		ZZ2608=ZZ2608+ZZ2608
+007610	755100		ZZ2608=ZZ2608+ZZ2608
+007610	732200		ZZ2608=ZZ2608+ZZ2608
+007610	664400		ZZ2608=ZZ2608+ZZ2608
+007610	000441		0 8192 -ZZ1608
+007611	664400		0 ZZ2608
-	 mark 7911, -219	/91 aqar
+007612	777110		ZZ2609=ZZ2609+ZZ2609
+007612	776220		ZZ2609=ZZ2609+ZZ2609
+007612	774440		ZZ2609=ZZ2609+ZZ2609
+007612	771100		ZZ2609=ZZ2609+ZZ2609
+007612	762200		ZZ2609=ZZ2609+ZZ2609
+007612	744400		ZZ2609=ZZ2609+ZZ2609
+007612	711000		ZZ2609=ZZ2609+ZZ2609
+007612	622000		ZZ2609=ZZ2609+ZZ2609
+007612	000431		0 8192 -ZZ1609
+007613	622000		0 ZZ2609
-	 mark 7919, 62		/ 6 pisc
+007614	000174		ZZ2610=ZZ2610+ZZ2610
+007614	000370		ZZ2610=ZZ2610+ZZ2610
+007614	000760		ZZ2610=ZZ2610+ZZ2610
+007614	001740		ZZ2610=ZZ2610+ZZ2610
+007614	003700		ZZ2610=ZZ2610+ZZ2610
+007614	007600		ZZ2610=ZZ2610+ZZ2610
+007614	017400		ZZ2610=ZZ2610+ZZ2610
+007614	037000		ZZ2610=ZZ2610+ZZ2610
+007614	000421		0 8192 -ZZ1610
+007615	037000		0 ZZ2610
-	 mark 7923, -222	/93 aqar
+007616	777102		ZZ2611=ZZ2611+ZZ2611
+007616	776204		ZZ2611=ZZ2611+ZZ2611
+007616	774410		ZZ2611=ZZ2611+ZZ2611
+007616	771020		ZZ2611=ZZ2611+ZZ2611
+007616	762040		ZZ2611=ZZ2611+ZZ2611
+007616	744100		ZZ2611=ZZ2611+ZZ2611
+007616	710200		ZZ2611=ZZ2611+ZZ2611
+007616	620400		ZZ2611=ZZ2611+ZZ2611
+007616	000415		0 8192 -ZZ1611
+007617	620400		0 ZZ2611
-	 mark 7952, -470	/98 aqar
+007620	776122		ZZ2612=ZZ2612+ZZ2612
+007620	774244		ZZ2612=ZZ2612+ZZ2612
+007620	770510		ZZ2612=ZZ2612+ZZ2612
+007620	761220		ZZ2612=ZZ2612+ZZ2612
+007620	742440		ZZ2612=ZZ2612+ZZ2612
+007620	705100		ZZ2612=ZZ2612+ZZ2612
+007620	612200		ZZ2612=ZZ2612+ZZ2612
+007620	424400		ZZ2612=ZZ2612+ZZ2612
+007620	000360		0 8192 -ZZ1612
+007621	424400		0 ZZ2612
-	 mark 7969, -482	/99 aqar
+007622	776072		ZZ2613=ZZ2613+ZZ2613
+007622	774164		ZZ2613=ZZ2613+ZZ2613
+007622	770350		ZZ2613=ZZ2613+ZZ2613
+007622	760720		ZZ2613=ZZ2613+ZZ2613
+007622	741640		ZZ2613=ZZ2613+ZZ2613
+007622	703500		ZZ2613=ZZ2613+ZZ2613
+007622	607200		ZZ2613=ZZ2613+ZZ2613
+007622	416400		ZZ2613=ZZ2613+ZZ2613
+007622	000337		0 8192 -ZZ1613
+007623	416400		0 ZZ2613
-	 mark 7975, 16		/ 8 pisc
+007624	000040		ZZ2614=ZZ2614+ZZ2614
+007624	000100		ZZ2614=ZZ2614+ZZ2614
+007624	000200		ZZ2614=ZZ2614+ZZ2614
+007624	000400		ZZ2614=ZZ2614+ZZ2614
+007624	001000		ZZ2614=ZZ2614+ZZ2614
+007624	002000		ZZ2614=ZZ2614+ZZ2614
+007624	004000		ZZ2614=ZZ2614+ZZ2614
+007624	010000		ZZ2614=ZZ2614+ZZ2614
+007624	000331		0 8192 -ZZ1614
+007625	010000		0 ZZ2614
-	 mark 7981, 133		/10 pisc
+007626	000412		ZZ2615=ZZ2615+ZZ2615
+007626	001024		ZZ2615=ZZ2615+ZZ2615
+007626	002050		ZZ2615=ZZ2615+ZZ2615
+007626	004120		ZZ2615=ZZ2615+ZZ2615
+007626	010240		ZZ2615=ZZ2615+ZZ2615
+007626	020500		ZZ2615=ZZ2615+ZZ2615
+007626	041200		ZZ2615=ZZ2615+ZZ2615
+007626	102400		ZZ2615=ZZ2615+ZZ2615
+007626	000323		0 8192 -ZZ1615
+007627	102400		0 ZZ2615
-	 mark 7988, 278		/70 pegs
+007630	001054		ZZ2616=ZZ2616+ZZ2616
+007630	002130		ZZ2616=ZZ2616+ZZ2616
+007630	004260		ZZ2616=ZZ2616+ZZ2616
+007630	010540		ZZ2616=ZZ2616+ZZ2616
+007630	021300		ZZ2616=ZZ2616+ZZ2616
+007630	042600		ZZ2616=ZZ2616+ZZ2616
+007630	105400		ZZ2616=ZZ2616+ZZ2616
+007630	213000		ZZ2616=ZZ2616+ZZ2616
+007630	000314		0 8192 -ZZ1616
+007631	213000		0 ZZ2616
-	 mark 8010, -489	/101 aqar
+007632	776054		ZZ2617=ZZ2617+ZZ2617
+007632	774130		ZZ2617=ZZ2617+ZZ2617
+007632	770260		ZZ2617=ZZ2617+ZZ2617
+007632	760540		ZZ2617=ZZ2617+ZZ2617
+007632	741300		ZZ2617=ZZ2617+ZZ2617
+007632	702600		ZZ2617=ZZ2617+ZZ2617
+007632	605400		ZZ2617=ZZ2617+ZZ2617
+007632	413000		ZZ2617=ZZ2617+ZZ2617
+007632	000266		0 8192 -ZZ1617
+007633	413000		0 ZZ2617
-	 mark 8049, 116		/17 pisc
+007634	000350		ZZ2618=ZZ2618+ZZ2618
+007634	000720		ZZ2618=ZZ2618+ZZ2618
+007634	001640		ZZ2618=ZZ2618+ZZ2618
+007634	003500		ZZ2618=ZZ2618+ZZ2618
+007634	007200		ZZ2618=ZZ2618+ZZ2618
+007634	016400		ZZ2618=ZZ2618+ZZ2618
+007634	035000		ZZ2618=ZZ2618+ZZ2618
+007634	072000		ZZ2618=ZZ2618+ZZ2618
+007634	000217		0 8192 -ZZ1618
+007635	072000		0 ZZ2618
-	 mark 8059, -418	/104 aqar
+007636	776272		ZZ2619=ZZ2619+ZZ2619
+007636	774564		ZZ2619=ZZ2619+ZZ2619
+007636	771350		ZZ2619=ZZ2619+ZZ2619
+007636	762720		ZZ2619=ZZ2619+ZZ2619
+007636	745640		ZZ2619=ZZ2619+ZZ2619
+007636	713500		ZZ2619=ZZ2619+ZZ2619
+007636	627200		ZZ2619=ZZ2619+ZZ2619
+007636	456400		ZZ2619=ZZ2619+ZZ2619
+007636	000205		0 8192 -ZZ1619
+007637	456400		0 ZZ2619
-	 mark 8061, 28		/18 pisc
+007640	000070		ZZ2620=ZZ2620+ZZ2620
+007640	000160		ZZ2620=ZZ2620+ZZ2620
+007640	000340		ZZ2620=ZZ2620+ZZ2620
+007640	000700		ZZ2620=ZZ2620+ZZ2620
+007640	001600		ZZ2620=ZZ2620+ZZ2620
+007640	003400		ZZ2620=ZZ2620+ZZ2620
+007640	007000		ZZ2620=ZZ2620+ZZ2620
+007640	016000		ZZ2620=ZZ2620+ZZ2620
+007640	000203		0 8192 -ZZ1620
+007641	016000		0 ZZ2620
-	 mark 8064, -344	/105 aqar
+007642	776516		ZZ2621=ZZ2621+ZZ2621
+007642	775234		ZZ2621=ZZ2621+ZZ2621
+007642	772470		ZZ2621=ZZ2621+ZZ2621
+007642	765160		ZZ2621=ZZ2621+ZZ2621
+007642	752340		ZZ2621=ZZ2621+ZZ2621
+007642	724700		ZZ2621=ZZ2621+ZZ2621
+007642	651600		ZZ2621=ZZ2621+ZZ2621
+007642	523400		ZZ2621=ZZ2621+ZZ2621
+007642	000200		0 8192 -ZZ1621
+007643	523400		0 ZZ2621
-	 mark 8159, 144		/28 pisc
+007644	000440		ZZ2622=ZZ2622+ZZ2622
+007644	001100		ZZ2622=ZZ2622+ZZ2622
+007644	002200		ZZ2622=ZZ2622+ZZ2622
+007644	004400		ZZ2622=ZZ2622+ZZ2622
+007644	011000		ZZ2622=ZZ2622+ZZ2622
+007644	022000		ZZ2622=ZZ2622+ZZ2622
+007644	044000		ZZ2622=ZZ2622+ZZ2622
+007644	110000		ZZ2622=ZZ2622+ZZ2622
+007644	000041		0 8192 -ZZ1622
+007645	110000		0 ZZ2622
-	 mark 8174, -149	/30 pisc
+007646	777324		ZZ2623=ZZ2623+ZZ2623
+007646	776650		ZZ2623=ZZ2623+ZZ2623
+007646	775520		ZZ2623=ZZ2623+ZZ2623
+007646	773240		ZZ2623=ZZ2623+ZZ2623
+007646	766500		ZZ2623=ZZ2623+ZZ2623
+007646	755200		ZZ2623=ZZ2623+ZZ2623
+007646	732400		ZZ2623=ZZ2623+ZZ2623
+007646	665000		ZZ2623=ZZ2623+ZZ2623
+007646	000022		0 8192 -ZZ1623
+007647	665000		0 ZZ2623
 007650		4q,
-	 mark 8188, -407	/ 2 ceti
+007650	776320		ZZ2624=ZZ2624+ZZ2624
+007650	774640		ZZ2624=ZZ2624+ZZ2624
+007650	771500		ZZ2624=ZZ2624+ZZ2624
+007650	763200		ZZ2624=ZZ2624+ZZ2624
+007650	746400		ZZ2624=ZZ2624+ZZ2624
+007650	715000		ZZ2624=ZZ2624+ZZ2624
+007650	632000		ZZ2624=ZZ2624+ZZ2624
+007650	464000		ZZ2624=ZZ2624+ZZ2624
+007650	000004		0 8192 -ZZ1624
+007651	464000		0 ZZ2624
 007652			 start 4
`
