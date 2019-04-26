// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): Delete this file once notary is available. See #30601.

package modfetch

var goSumPin = `
cloud.google.com/go
 v0.29.0 h1:gv/9Wwq5WPVIGaROMQg8tw4jLFFiyacODxEIrlz0wTw= h1:aQUYkXzVsufM+DwF1aE+0xfcU+56JwCaLick0ClmMTw=
 v0.30.0 h1:xKvyLgk56d0nksWq49J0UyGEeUIicTl4+UBiX1NPX9g= -
 v0.31.0 h1:o9K5MWWt2wk+d9jkGn2DAZ7Q9nUdnFLOpK9eIkDwONQ= -
 v0.32.0 h1:DSt59WoyNcfAInilEpfvm2ugq8zvNyaHAm9MkzOwRQ4= -
 v0.33.0 h1:1kNZapR5iXMPsPEca6Rqg+EN4/8/ZukNjMdwNQEllWk= -
 v0.33.1 h1:fmJQWZ1w9PGkHR1YL/P7HloDvqlmKQ4Vpb7PC2e+aCk= -
 v0.34.0 h1:eOI3/cP2VTU6uZLDYAoic+eyzzB9YyGmJ7eIjl8rOPg= -
 v0.35.0 h1:+ZrbIJ3Qm81r3IU2+/ueWMpAMXLF3Nwy2dF7NkBdEXk= h1:UE4juzxiHpKLbqrOrwVrKuaZvUtLA9CSnaYO+y53jxA=
 v0.35.1 h1:LMe/Btq0Eijsc97JyBwMc0KMXOe0orqAMdg7/EkywN8= h1:wfjPZNvXCBYESy3fIynybskMP48KVPrjSPCnXiK7Prg=
 v0.36.0 h1:+aCSj7tOo2LODWVEuZDZeGCckdt6MlSF+X/rB3wUiS8= h1:RUoy9p/M4ge0HzT8L+SDZ8jg+Q6fth0CiBuhFJpSV40=
code.cloudfoundry.org/lager
 v1.0.0 h1:ZW/aJB8upEKcCxUexFLoVjT32Iex3eWdkjw0F7wHbpE= h1:O2sS7gKP3HM2iemG+EnwvyNQK7pTSC6Foi4QiMp9sSk=
 v1.1.0 h1:v0RELJ2jqTeF2DW7PNjZaaGlrXbVxJBVz3uLxdP3fuY= -
 v2.0.0+incompatible h1:WZwDKDB2PLd/oL+USK4b4aEjUymIej9My2nUQ9oWEwQ= -
collectd.org
 v0.1.0 h1:zWj1YPi6zkzxTsu/j1pGJgjhghXx0pO2pSefsOf4B1k= h1:A/8DzQBkF6abtvrT2j/AU/4tiBgJWYyh0y/oB/4MlWE=
 v0.2.0 h1:49s4ZrBFMn32+doAe5Y+GMFZH0L0P9Z+65roMnMmE1g= -
 v0.3.0 h1:iNBHGw1VvPJxH2B6RiFWFZ+vsjo1lCdRszBeOuwGi00= -
git.apache.org/thrift.git
 v0.12.0 h1:CMxsZlAmxKs+VAZMlDDL0wXciMblJcutQbEe3A9CYUM= h1:fPE2ZNJGynbRyZ4dJvy6G277gSllfV2HJqblrnkyeyg=
github.com/ActiveState/tail
 v1.0.0 h1:awEa/oIeyIkiGmgtK/pn9UssITMOh4jcx2hcOkmZtgk= h1:8bqcJf9F0fOp1g3E3BKmY49fyWF549IlCWHQx5OTrtY=
github.com/Azure/azure-sdk-for-go
 v22.1.0+incompatible h1:9DQsYsbAliwUGtAXF4i/u517yPlGKW2VLJEa3R/asck= h1:9XXNKU+eRnpl9moKnB4QOLf1HestfXbmab5FXxiDBjc=
 v22.1.1+incompatible h1:Nm5K4x9E1lbBWAol0ROVwcmrc0vOC+8wyEuJAFTum4g= -
 v22.2.2+incompatible h1:dnM65i68vx79S5ugocLMoJB6As2U1IXxa995LdjIQ28= -
 v23.0.0+incompatible h1:eFpwpmS1ZEEOd7X3q9GTFqrXAspzirOUPtrzxnzfWBA= -
 v23.1.0+incompatible h1:g0myUjVaQlg8xQ0T5ucdiw8PDxMF3UJBiLm0grc4e0E= -
 v23.2.0+incompatible h1:bch1RS060vGpHpY3zvQDV4rOiRw25J1zmR/B9a76aSA= -
 v24.0.0+incompatible h1:GdF0ozHojCPSZH1LPWA2+XHQ3G/mapn0G+PCIlMVZg4= -
 v24.1.0+incompatible h1:P7GocB7bhkyGbRL1tCy0m9FDqb1V/dqssch3jZieUHk= -
 v25.0.0+incompatible h1:kVuVjZDTvVc1bj6RWhB/08kTrEgz3xDDzZq5sWYb5d4= -
 v25.1.0+incompatible h1:bA8mqsHUc9RbzHG64A6r7KnpvLFHJdxrpI75FrFln2M= -
github.com/Azure/go-autorest
 v11.2.4+incompatible h1:Xh0rJxHAuwtFOaMTA3sDAZkHM+2vWyKR4UhwLo6WOfg= h1:r+4oMnoxhatjLLJ6zxSWATqVooLgysK6ZNox3g/xq24=
 v11.2.5+incompatible h1:PplSl6LVDNWUb/BLfhYsUA5unZ6qULiW2KB/v2Qw8tI= -
 v11.2.6+incompatible h1:YIFRvuc6ECAtClY0I91zGjnv4y8Nf+1XwwjGNsbUJ/k= -
 v11.2.7+incompatible h1:DQRVSOFe2EiYoS/FZgwtjdRnHwuk+HvEXp8PEBmFH7w= -
 v11.2.8+incompatible h1:Q2feRPMlcfVcqz3pF87PJzkm5lZrL+x6BDtzhODzNJM= -
 v11.3.0+incompatible h1:oPIb2R8fwU91NsavCEqDYBLTUTcxckx5kIRHY7zBi/E= -
 v11.3.1+incompatible h1:Pzn7+3iKqV1UAbwKarPKc4asZMJe9fQvs0csgYl6p4A= -
 v11.3.2+incompatible h1:2bRmoaLvtIXW5uWpZVoIkc0C1z7c84rVGnP+3mpyCRg= -
 v11.4.0+incompatible h1:z3Yr6KYqs0nhSNwqGXEBpWK977hxVqsLv2n9PVYcixY= -
 v11.5.0+incompatible h1:zp9GQJhEX+EBqEYC2MEGQ+gjKFEPRAWtfwcmstS2hGk= -
github.com/BurntSushi/toml
 v0.1.0 h1:o5kUUOTNGTDgVf5Cmj6EkHCBf1pZ/OUs+wdQruDOSAI= h1:xHWCNGjB5oqiDr8zfno3MHue2Ht5sIBksp03qcyfWMU=
 v0.2.0 h1:OthAm9ZSUx4uAmn3WbPwc06nowWrByRwBsYRhbmFjBs= -
 v0.3.0 h1:e1/Ivsx3Z0FVTV0NSOv/aVgbUWyQuzj7DDnFblkRvsY= -
 v0.3.1 h1:WXkYYl6Yr3qBf1K79EBnL4mak0OimBfB0XUf9Vl28OQ= -
github.com/ChimeraCoder/anaconda
 v1.0.0 h1:B7KZV+CE2iwbC15sh+rh5vaWs4+XJx1XC4iHvHtsZrQ= h1:TCt3MijIq3Qqo9SBtuW/rrM4x7rDfWqYWHj8T7hLcLg=
 v2.0.0+incompatible h1:F0eD7CHXieZ+VLboCD5UAqCeAzJZxcr90zSCcuJopJs= -
github.com/Jeffail/gabs
 v0.9.0 h1:WY/QB2yjqNneqvu5WX0bVgq0WvKFHCzqJYcfv/ZDRkY= h1:6xMvQMK4k33lb7GUUpaAPh6nKMmemQeg5d4gn7/bOXc=
 v1.0.0 h1:yGg0yih4Q4ZSbiixE3C7a2wCXbc8scritLhs19duATw= -
 v1.1.0 h1:kw5zCcl9tlJNHTDme7qbi21fDHZmXrnjMoXos3Jw/NI= -
 v1.1.1 h1:V0uzR08Hj22EX8+8QMhyI9sX2hwRu+/RJhJUmnwda/E= -
 v1.2.0 h1:uFhoIVTtsX7hV2RxNgWad8gMU+8OJdzFbOathJdhD3o= -
github.com/Masterminds/cookoo
 v1.0.0 h1:KcHRuuRx3qy6ejolMDOCUSNSC4s690NFky8Z3lcQIgU= h1:oXuAk0dniDrcAwWZQYMWYzJnpnAZqQFcR7CkHsEIBHc=
 v1.1.0 h1:gdXvuBqtCtrZKbwhRLQQCBZ+fzZvo7/fNuCoewPcLGc= -
 v1.2.0 h1:3/dKnvddvxmuJgSPOWbefSIKauVdiDWfGx8sv0YnK6o= -
 v1.3.0 h1:zwplWkfGEd4NxiL0iZHh5Jh1o25SUJTKWLfv2FkXh6o= -
github.com/Masterminds/glide
 v0.11.0 h1:IJUPyi5o1aB9l6udviKVnR0FiQTT+MpZfFMvyOjl8fg= h1:STyF5vcenH/rUqTEv+/hBXlSTo7KYwg2oc2f4tzPWic=
 v0.11.1 h1:L9cLkidcHlrJLw5AFnSzFQ+tyylhQAihJ47NclXIPqU= -
 v0.12.0 h1:WYiatWlWzzeAJMvr1Js5aiH26OgOtwAhCCJDSX0eIUA= -
 v0.12.1 h1:zKYH5xKi2dSc1reJhXFBVKiMpznAY3r20sVO2/zrC0A= -
 v0.12.2 h1:szgrhFbKtri+N+c/xfN4JbKYPG3mKTKElg3CWXS8Mjs= -
 v0.12.3 h1:QkcpAwXPCoYq9a9C3U6UtivvYz/RXSvUY/Ob+ikKOBQ= -
 v0.13.0 h1:X5U6GJXcFQ2HgCZeCHIlCMJUfEUzFf5DhHd7lJ7gWJI= -
 v0.13.1 h1:NhvMmo3LUDtFB4d27wz0PCtgtHpyllDeqqog4KEXy3E= -
 v0.13.2 h1:M5MOH04TyRiMBVeWHbifqTpnauxWINIubTCOkhXh+2g= -
github.com/Masterminds/semver
 v1.1.1 h1:zaTaSIIZiw2DIR1ZvGd3Drz9UMzMxkPrIwOvOdJddFw= h1:MB6lktGJrhw8PrUyiEoblNEGEQ+RzHPF078ddwwvV3Y=
 v1.2.0 h1:Jo7YVdCWcBIefBe13vUNwemam8K3vwuNJfLa1oRnZps= -
 v1.2.1 h1:voGDv5f8R16RnjK9BrMTmvYIqQCs6Jvcsfqzh2lNHzE= -
 v1.2.2 h1:ptelpryog9A0pR4TGFvIAvw2c8SaNrYkFtfrxhSviss= -
 v1.2.3 h1:FZV+FKRA+7mSfv17UZNtqT4oRSrv2LMDi8MDlkDI8lM= -
 v1.3.0 h1:7H8mLwaeisxNSFxW39uQ9UHGv7HOevcDtjFjgbPDE/4= -
 v1.3.1 h1:4CEBDLZtuloRJFiIzzlR/VcQOCiFzhaaa7hE4DEB97Y= -
 v1.4.0 h1:h9TTGlRMRjil1wsQTRj3+COyXMngVsuK6HMph+Ma7ds= -
 v1.4.1 h1:CaDA1wAoM3rj9sAFyyZP37LloExUzxFGYt+DqJ870JA= -
 v1.4.2 h1:WBLTQ37jOCzSLtXNdoo8bNM8876KhNqOKvrlGITgsTc= -
github.com/Masterminds/sprig
 v2.13.0+incompatible h1:bpkP6O4TFdP4u0qL/7B2LWe6uobYLUgP6Hfh2b8AdGg= h1:y6hNFY5UBTIWBxnzTeuNhlNS5hqE0NB0E6fgfo2Br3o=
 v2.14.0+incompatible h1:nC0fgY6y1UohQeh3PIaXh2IY54dS8JaW2JZwvJM+3ts= -
 v2.14.1+incompatible h1:rTHERm50Xp1Cbb8x7xBCeDp//jMMqqR44EWw7KwSXUQ= -
 v2.15.0+incompatible h1:0gSxPGWS9PAr7U2NsQ2YQg6juRDINkUyuvbb4b2Xm8w= -
 v2.16.0+incompatible h1:QZbMUPxRQ50EKAq3LFMnxddMu88/EUUG3qmxwtDmPsY= -
 v2.17.0+incompatible h1:GZlRWk/aIMk27GWitLvMubJT9qpx5r9Y3312igRSnNo= -
 v2.17.1+incompatible h1:PChbxFGKTWsg9IWh+pSZRCSj3zQkVpL6Hd9uWsFwxtc= -
 v2.18.0+incompatible h1:QoGhlbC6pter1jxKnjMFxT8EqsLuDE6FEcNbWEpw+lI= -
github.com/Masterminds/squirrel
 v1.1.0 h1:baP1qLdoQCeTw3ifCdOq2dkYc6vGcmRdaociKLbEJXs= h1:yaPeOnPG5ZRwL9oKdTsO/prlkPbXWZlRVMQ/gGlzIuA=
github.com/Masterminds/vcs
 v1.8.0 h1:/n3MgEhtRfZQ/6yw651aAlYFrxV5RS+BYm0bL0R8isE= h1:N09YCmOQr6RLxC6UNHzuVwAdodYbbnycGHSmwVJjcKA=
 v1.9.0 h1:I+px62qwzoQwQW/sZF6MbwXDFPebpNer8OFTrrkeEHY= -
 v1.10.0 h1:jkK5Cc7D15F+IeYxzaLLnwjIJktoVWmop57/MW4Ofqg= -
 v1.10.1 h1:qfaxFNhilVumtLivRl3XwEdvXhb5fBzRh2RcsNJZMQg= -
 v1.10.2 h1:hTQV/PE/IuJK+ZmaI3DX0tqVHirXAZ/6S5BHSAQAuHo= -
 v1.11.0 h1:wvFGFdyFW2AviPigHWPbRsUJBn8pZWiAGSNVqiX84IE= -
 v1.11.1 h1:JWGoxi1Ex/YnNqXE8IWCabkKGihSkFxwDKcM8ojzY7w= -
 v1.12.0 h1:bt9Hb4XlfmEfLnVA0MVz2NO0GFuMN5vX8iOWW38Xde4= -
github.com/Microsoft/go-winio
 v0.4.2 h1:ZJbCAgklAxf91aYmsiUHS7dIGCqw+EjpOg/oq/Groaw= h1:VhR8bwka0BXejwEJY73c50VrPtXAaKcyvVC4A4RozmA=
 v0.4.3 h1:M3NHMuPgMSUPdE5epwNUHlRPSVzHs8HpRTrVXhR0myo= -
 v0.4.4 h1:O2b99gLN+3goL4SB45jFjq0reF0AVDt9d5wU/kInBt4= -
 v0.4.5 h1:U2XsGR5dBg1yzwSEJoP2dE2/aAXpmad+CNG2hE9Pd5k= -
 v0.4.6 h1:Tu8dlnF1wvUKKqr011GFneCoyIn7D+Q2uq6AKmQnGrA= -
 v0.4.7 h1:vOvDiY/F1avSWlCWiKJjdYKz2jVjTK3pWPHndeG4OAY= -
 v0.4.8 h1:1TfLnrRLKLEVn9t+5FpZqiEJkzpm1QDVs4rZZlUpFOk= -
 v0.4.9 h1:3RbgqgGVqmcpbOiwrjbVtDHLlJBGF6aE+yHmNtBNsFQ= -
 v0.4.10 h1:NrhPZI+cp3Fjmm5t/PZkVuir43JIRLZG/PSKK7atSfw= -
 v0.4.11 h1:zoIOcVf0xPN1tnMVbTtEdI+P8OofVk3NObnwOQ6nK2Q= -
github.com/Microsoft/hcsshim
 v0.7.12 h1:VCjS2UYlYyMfRnCus+yhbJZBi9DeFSMBKrggG/PAeHk= h1:Op3hHsoHPAvb6lceZHDtd9OkTew38wNoXnJs8iY7rUg=
 v0.7.13 h1:GHTF675XCwX4e6eezaNXE643Tiqn8ulQtSmWXP1r5pw= -
 v0.7.14 h1:T/ZNh+dsrD5XMH6dI94hapFiz2JYEu8WDW1d6zZvHBI= -
 v0.8.0 h1:4octbSGAQCm/By5owYT5RUGzg/tWZwTaFDeWM1eW7q4= -
 v0.8.1 h1:0RKPd1pQB/4YRjdw0jFwq3A5nWFN4n1ojNzcm4B+8ZI= -
 v0.8.2 h1:kfFBcPjZvdu+4yP1KBHszFKyM7uk86FujV3bq7UlIxk= -
 v0.8.3 h1:KWCdVGOju81E0RL4ndn9/E6I4qMBi6kuPw1W4yBYlCw= -
 v0.8.4 h1:BxoCMvp9PlnwkqrgJC4wN7Y0b4TeuK1DZTOmubzFvz4= -
 v0.8.5 h1:kg/pore5Yyf4DXQ5nelSqfaYQG54YIdNeFRKJaPnFiM= -
 v0.8.6 h1:ZfF0+zZeYdzMIVMZHKtDKJvLHj76XCuVae/jNkjj0IA= -
github.com/NYTimes/gziphandler
 v1.0.0 h1:OswZCvpiFsNRCbeapdJxDuikAqVXTgV7XAht8S9olZo= h1:3wb06e3pkSAbeQ52E9H9iFoQsEEwGN64994WTCIhntQ=
 v1.0.1 h1:iLrQrdwjDd52kHDA5op2UBJFjmOb9g+7scBan4RN8F0= -
 v1.1.0 h1:wkMjq4kSz11Zer+ncYWNBQDlj9Y5RLloY/Tb8yOj6gA= h1:EwmLXLwj3Rvq6vawd3hKEPUcQRyz2CDE1bov6dy8HNQ=
github.com/Pallinder/go-randomdata
 v1.1.0 h1:gUubB1IEUliFmzjqjhf+bgkg1o6uoFIkRsP3VrhEcx8= h1:yHmJgulpD2Nfrm0cR9tI/+oAgRqCQQixsA8HyRZfV9Y=
github.com/PuerkitoBio/goquery
 v0.3.2 h1:DFIz6qk2ErBE+73SrW6YaiGywIaScwhk+hnRJeuQTBg= h1:T9ezsOHcCrDCgA8aF1Cqr3sSYbO/xgdy8/R/XiIMAhA=
 v1.0.0 h1:jZDjai9V7bAxImA/boIynYuY1tbp/rysh4Xsr/L4uiw= -
 v1.0.1 h1:pb5HCQuIF/QWUtO3ZivkPuN0kC+DE8YbO0/+kSRy8Qk= -
 v1.0.2 h1:6eVgli+CgrpInQgyW5Unj3aqfzqFk/ALcKm6m0w7hgA= -
 v1.1.0 h1:QUDKATbrxlrC/VtTGXPgSF28dtBbniZz0X2sp/Twom4= -
 v1.2.0 h1:Ej6nIAQZhMyRPNV5jOVZlE3XY4YsPYrppZXL6G9jza0= -
 v1.3.0 h1:2LzdaeRwZjIMW7iKEei51jiCPB33mou4AI7QCzS4NgE= -
 v1.4.0 h1:13fV4AYmaSopdNp8KWDUlLyU5INklBkYk0tsTfxRO2U= -
 v1.4.1 h1:smcIRGdYm/w7JSbcdeLHEMzxmsBQvl8lhf0dSw2nzMI= -
 v1.5.0 h1:uGvmFXOA73IKluu/F84Xd1tt/z07GYm8X49XKHP7EJk= h1:qD2PgZ9lccMbQlc7eEOjaeRlFQON7xY8kdmcsrnKqMg=
github.com/PuerkitoBio/purell
 v0.1.0 h1:N8Bcc53nei5frgNYgAKo93qMUVdU5LUGHCBv8efdVcM= h1:c11w/QuzBsJSee3cPx9rAFu61PvFxuPbtSwDGJws/X0=
 v1.0.0 h1:0GoNN3taZV6QI81IXgCbxMyEaJDXMSIjArYBCYzVVvs= -
 v1.1.0 h1:rmGxhojJlM0tuKtfdvliR84CFHljx9ag64t2xmVkjK4= -
 v1.1.1 h1:WEQqlqaGbrPkxLJWfBwQmfEAE1Z7ONdDLqrN38tNFfI= -
github.com/RangelReale/osin
 v1.0.0 h1:mkPOQ2FdespabcwRi3/1X5/a/7FwOgfguWFR+1PA9L8= h1:k/PH1SjZDitJDtK3zHm/XZRi+bRz6i3rhx9qE9p54CY=
 v1.0.1 h1:JcqBe8ljQq9WQJPtioXGxBWyIcfuVMw0BX6yJ9E4HKw= -
github.com/SeanDolphin/bqschema
 v1.0.0 h1:iCYFd5Qsw6caM2k5/SsITSL9+3kQCr+oz6pnNjWTq90= h1:TYInVncsPIZH7kybQoIUNJ4pFX1cUc8LoP9RSOxIs6c=
github.com/SermoDigital/jose
 v0.9.1 h1:atYaHPD3lPICcbK1owly3aPm0iaJGSGPi0WD4vLznv8= h1:ARgCUhI1MHQH+ONky/PAtmVHQrP5JlGY0F3poXOp/fA=
github.com/Shopify/sarama
 v1.12.0 h1:SGfRgQ8Qq7DfnoAzGEQDssnqz5ZHl7cmpzpJKLj3UwQ= h1:FVkBWblsNy7DGZRfXLU0O9RCGt5g3g3yEuWXgklEdEo=
 v1.13.0 h1:R+4WFsmMzUxN2uiGzWXoY9apBAQnARC+B+wYvy/kC3k= -
 v1.14.0 h1:ybE26/v5eppjkQZmMAttQK8lFiNYnk/aWYVU/IgmWpg= -
 v1.15.0 h1:v/Q3THMtunYfvKhbFfhegInfoW70HoNgsOdmuvFN5Qg= -
 v1.16.0 h1:9pI5+ZN06jB3bu5kHXqzzaErMC5rimcIZBQL9IOiEQ0= -
 v1.17.0 h1:Y2/FBwElFVwt7aLKL3fDG6hh+rrlywR6uLgTgKObwTc= -
 v1.18.0 h1:Ha2FAOngREft7C44ouUXDxSZ/Y/77IDCMV1YS4AnUkI= -
 v1.19.0 h1:9oksLxC6uxVPHPVYUmq6xhr1BOF/hHobWH2UzO67z1s= -
 v1.20.0 h1:wAMHhl1lGRlobeoV/xOKpbqD2OQsOvY4A/vIOGroIe8= -
 v1.20.1 h1:Bb0h3I++r4eX333Y0uZV2vwUXepJbt6ig05TUU1qt9I= -
github.com/Terry-Mao/gopush-cluster
 v1.0.0 h1:hVyXf8fLoQ3a5fOykdIyCcUbmUs4kdKDa97hwrCQw3o= h1:qU0tnOO1fIrsB1jbQSY/YfeS25dl6q3Bb8ucul3spuc=
 v1.0.1 h1:7d5fQgFxOI1ej0eL2b1TuuPu7nNdmi4xD434/zfqXt0= -
 v1.0.2 h1:O8yRpflCTtqiMR/VNYmrElTo2U8m2dqHLVQ/wOzLc70= -
 v1.0.3 h1:ryRDuFPra6L04k/UJkuxqlCpsnWw4SrBGeOv8X0WxYo= -
 v1.0.4 h1:Wc+VxOXW8wtJjYWCcLneNqNuaUP/HcYjZm5JeMLmCBE= -
github.com/VividCortex/ewma
 v1.1.1 h1:MnEK4VOv6n0RSY4vtRe3h11qjxL3+t0B8yOL8iMXdcM= h1:2Tkkvm3sRDVXaiyucHiACn4cqf7DpdyLvmxzcbUokwA=
github.com/Workiva/go-datastructures
 v1.0.41 h1:DsFvBLOojxUVOXpxMaFRMgrszuLN6+O/J96O3rTaQt8= h1:Z+F2Rca0qCsVYDS8z7bAGm8f3UkzuWYS/oBZz5a7VVA=
 v1.0.42 h1:yf94hF8U/DGDH/J4I5iYKvYQusHB9OZpNVJjU4jROKM= -
 v1.0.43 h1:PAzvm/sZzqX50iy+LF+sNKgDCIFZPDPDNMOha7nY7Po= -
 v1.0.44 h1:zNyQ5b0vc5aqzudAxCD2HFeNNpZLKCQCqP/D4HA54GA= -
 v1.0.45 h1:vq8+9mcimFV5UXJCTMxOoaPMcaE9gztoc5x1UbaRJUc= -
 v1.0.46 h1:85Vm9guMMJ+/+Ns23WrjK7bDE67uos7xmWHThvxaru4= -
 v1.0.47 h1:5dcz+D13KFP0F46P5xoa4xmmcUcbimzJm7ryCI8MFsE= -
 v1.0.48 h1:e/we+zYmL9Bro7bAAnS5xWmjg5qc0y5G+TOS2ZctW0o= -
 v1.0.49 h1:cKU4n0/psXU6GDjK4kpJU2koN7BZ0klqotTMJ1Rbj2s= -
 v1.0.50 h1:slDmfW6KCHcC7U+LP3DDBbm4fqTwZGn1beOFPfGaLvo= -
github.com/abbot/go-http-auth
 v0.4.0 h1:QjmvZ5gSC7jm3Zg54DqWE/T5m1t2AfDu6QlXJT0EVT0= h1:Cz6ARTIzApMJDzh5bRMSUou6UMSp0IEXg9km/ci7TJM=
github.com/aerospike/aerospike-client-go
 v1.32.0 h1:0PE+aQUqQ1ATb3k1y0UvEC8scCwg9ksq9LKXvoYA2uY= h1:zj8LBEnWBDOVEIJt8LvaRvDG5ARAoa5dBeHaB472NRc=
 v1.33.0 h1:xZ1sTMKizie136jqs86fpOJ/P9IJSQt9H2YmxyOi0lk= -
 v1.34.0 h1:sn6zvlwDls/NdIf1wCu1kVoYZ2iOHcwwqwcHP8Puef8= -
 v1.34.1 h1:QjKc40tj/4RBvgFuWOb0aF0WE/E4MEnaGO/CahG2o90= -
 v1.34.2 h1:zrlhAi7/HQbpde4undhtDt89rx9OJcyEi5KuDjnfIlg= -
 v1.35.0 h1:h507B7Z7SWjkL9Isb330BBxHD7YWN2EwXtb7ZtcFU3M= -
 v1.35.1 h1:iZhuKj6AKDFB4CZKLZk5YnkKWwCsdR2H8dK+FuU7uFg= -
 v1.35.2 h1:TWV2Bn59Ig7SM4Zue84fFsPGlfFJX/6xbuGHyYFS/ag= -
 v1.36.0 h1:EePkIW4FtF09vNJZqOSz7mx23069wOkPm4LmDF8CPB4= -
 v1.37.0 h1:+hbk5t1mtBfErFTxicGfRJVwxKjupzbgogaiR1iq/xY= -
github.com/agtorre/gocolorize
 v1.0.0 h1:TvGQd+fAqWQlDjQxSKe//Y6RaxK+RHpEU9X/zPmHW50= h1:cH6imfTkHVBRJhSOeSeEZhB4zqEYSq0sXuIyehgZMIY=
github.com/alecthomas/kingpin
 v2.1.9+incompatible h1:ZTeAUKWQRJHTb4dfsSD3Btl6FkY0kankelUrGpLoP2k= h1:59OFYbFVLKQKq+mqrL6Rw5bR0c3ACQaawgXx0QYndlE=
 v2.1.10+incompatible h1:UuHmdc/3On+EqxEqwqDPGMGxnDoVF7obItol51l7Ir8= -
 v2.1.11+incompatible h1:R7E1WEE7GddTW3qAhFWpGvNLgpRxM2cd8sQ7W+lRsJs= -
 v2.2.0+incompatible h1:9Pmf4VXakLfUqrRh76V27GNBHFUtEtSy5DaR3YqYbRE= -
 v2.2.1+incompatible h1:PjbMdYIZp9VKeasxWfpE4PX4QAatdbxldB5AOuAkSp0= -
 v2.2.2+incompatible h1:BvD/akJsxNMaKFmrFB3qsUs5sZSkdpIhSLWk88q40gQ= -
 v2.2.3+incompatible h1:4Wohd7Da/I0OWc3cvR93azRPNGwV5YaIax/kS0ttbgE= -
 v2.2.4+incompatible h1:NbnCKzqpYim2hHnyWx9RxeUwqN4L9OZrRsSFkykSzVg= -
 v2.2.5+incompatible h1:umWl1NNd72+ZvRti3T9C0SYean2hPZ7ZhxU8bsgc9BQ= -
 v2.2.6+incompatible h1:5svnBTFgJjZvGKyYBtMB0+m5wvrbUHiqye8wRJMlnYI= -
github.com/alexflint/go-arg
 v1.0.0 h1:VWNnY3DyBHiq5lcwY2FlCE5t5qyHNV0o5i1bkCIHprU= h1:Cto8k5VtkP4pp0EXiWD4ZJMFOOinZ38ggVcQ/6CGuRI=
github.com/anacrolix/torrent
 v1.0.0 h1:MxNBr5lKDK/fRtvAYPOzGGQrpj+70Lh7E5EVjzIIbXg= h1:N6lCILah/qQCk/gVjHsOnqaug7a5DqQyryCbZD1L188=
 v1.0.1 h1:YsjmBdyIUGDmbFEovln/NOiU+elyG92lcb2KyAiTcgE= h1:ZYV1Z2Wx3jXYSh26mDvneAbk8XIUxfvoVil2GW962zY=
github.com/andybalholm/cascadia
 v1.0.0 h1:hOCXnnZ5A+3eVDX8pvgl4kofXv2ELss0bKcqRySc45o= h1:GsXiBklL0woXo1j/WYWtSYYC4ouU9PqHO0sqidkEA4Y=
github.com/ant0ine/go-json-rest
 v2.0.4+incompatible h1:s+BBj3hMrPcnQ5IR/JqtHSyKMBx22RtpdcBhs+sqfjc= h1:q6aCt0GfU6LhpBsnZ/2U+mwe+0XB5WStbmwyoPfc+sk=
 v2.0.5+incompatible h1:rWfQkmeWC4q4asLRGCKrUX94l7cpT0x3Yd6yG3qc7uc= -
 v2.0.6+incompatible h1:W6jOYA9XWzg1YbZuso0tX4LqvrvM2BpJ4m/g9gQYbRc= -
 v2.1.0+incompatible h1:EqS+heLqxXPmunoL3cZHhGhYG/+eaVwhWlrhr+NKpYs= -
 v3.0.0+incompatible h1:+peIEe3YSRJqj7/TtqVVvkwXSzlgku2wsv29S/mb4F4= -
 v3.1.0+incompatible h1:grVLbQyVuPtQTP53i3qoE5eUBoE35CQe4EXKKP7Pn7c= -
 v3.2.0+incompatible h1:GiVzVzckqQwfSkJkt7YFKR+oBcjeCTvu824nD1xIEPg= -
 v3.3.0+incompatible h1:87hhMna0tJf0poWLrQMmu+EAjRsAmV6ZDXpzdTNKZX8= -
 v3.3.1+incompatible h1:SjCeJsKsU2vVk6Dx7IX+85/dp+mmivw38c+P21aiS4M= -
 v3.3.2+incompatible h1:nBixrkLFiDNAW0hauKDLc8yJI6XfrQumWvytE1Hk14E= -
github.com/antonholmquist/jason
 v1.0.0 h1:Ytg94Bcf1Bfi965K2q0s22mig/n4eGqEij/atENBhA0= h1:+GxMEKI0Va2U8h3os6oiUAetHAlGMvxjdpAH/9uvUMA=
github.com/apache/thrift
 v0.12.0 h1:pODnxUFNcjP9UTLZGTdeh+j16A8lJbRvD3rOtrk/7bs= h1:cp2SuWMxlEZw2r+iP2GNCdIi4C1qmUzdZFSVb+bacwQ=
github.com/apcera/nats
 v1.0.9 h1:EQkYBlS/flQLLLK5RQHom10zPwJdkpCvvGhKIRfkJ74= h1:EVgYOJsx6L49WUJRKtQ++1ALZYo1IKwVEcChTBIzEEQ=
 v1.1.2 h1:MNBsB1FHjZjHYczQqlp8g9BtKSAEkZT4Q/5gFrwUkiY= -
 v1.1.6 h1:Jk98/6ON9hljPsrlDAOKe7hJuuhKM9z8WTifuQoeWvY= -
 v1.2.0 h1:dVP26kUrGEoXIFCPZma/CYC0k8nXX5Sxoxw/08oHK+k= -
 v1.2.2 h1:gtb+n9B8VfxJ7rNYRCEkAeXddU2NvNV1u82qGZv5W5U= -
 v1.3.0 h1:QVEklKiLJ2QfKpBR6ou4/TgesXtnYLylZ/f0dOpvqI8= -
 v1.4.0 h1:GE7lrGfiRIK1HzMr8nXvpMdNzeXQCYx4D3bVYPAcOwQ= -
 v1.5.0 h1:ZhkjtgZ39OU8pxENsZ5lHTPjUDqifLMA3nVzF8bf9T0= -
 v1.6.0 h1:z9XGt8apN1127Hn3BoQqvTb7whzLem0NzCfjL8oZxiQ= -
 v1.7.0 h1:2u2xXtXdg9C+wiELz5rtheY1Djwukjsk6ePoIKL9gm0= -
github.com/apex/apex
 v0.11.0 h1:5e83w49hAYZ5TWu9bBCi0E9cJIW27tnkRXpGsNX9++w= h1:kof13vptkdeeeQe+P1xgCKzHCq2L7/Il8QM1HfjI7qY=
 v0.12.0 h1:SHitRgxR3QlaRwVRNGkPR4yb3gFY1899ri09ErGorjw= -
 v0.13.0 h1:3CB6fXl5USHtBME0X9GjwpBwY0H0UFplMhyNxGLqbuw= -
 v0.13.1 h1:URS3EYZwkBzJxvjKkN/h21n2rJ22bUthBAcuBnQRXDs= -
 v0.14.0 h1:/ENBKpIhCdCCLTIQs+2H3up6em1XEpWYYY4pjLMtnRM= -
 v0.15.0 h1:UhejXfxrJjnpTBeH3fjoNtyNR1hWOi50v89BtAFInqs= -
 v0.16.0 h1:tpByjslO1YQw7VxGFAGgnGjVFxdda9aT3GucB0j4rwc= -
 v1.0.0-rc1 h1:UKSrvT5J+MYPoxxWd/GSESU9K+DcBj2RyHGs7OQxxIg= -
 v1.0.0-rc2 h1:Xy536Df23zqGZjns/pI6KdqJ5JsJvVOB2I83Z9AR/EQ= -
 v1.0.0-rc3 h1:s/qSTZ5niFriVO8T7aD8FGtF8WTxwGRoqdbR3QavGIM= -
github.com/apex/log
 v1.0.0 h1:5UWeZC54mWVtOGSCjtuvDPgY/o0QxmjQgvYZ27pLVGQ= h1:yA770aXIDQrhVOIGurT/pVdfCpSq1GQV/auzMN5fzvY=
 v1.1.0 h1:J5rld6WVFi6NxA6m8GJ1LJqu3+GiTFIt3mYv27gdQWI= -
github.com/apparentlymart/go-cidr
 v1.0.0 h1:lGDvXx8Lv9QHjrAVP7jyzleG4F9+FkRhJcEsDFxeb8w= h1:EBcsNrHc3zQeuaeCeCtQruQm+n9/YjEn/vI25Lg7Gwc=
github.com/apparentlymart/go-rundeck-api
 v0.0.1 h1:U3B4JDw4McLlsZ/1UlTjTMnM0SjqO03svfvu7qRPA7g= h1:U6OjNHcY3edY04ILn+KNrWZm3j15cPzW7PyjtIQOh1Y=
github.com/appc/cni
 v0.4.0 h1:5jSFAp0e3pMhFCqKq6uwJUbTvpONtyjc1vZsHEispzs= h1:+JvcwZORvjAIdTtOBUNnR7Ry5kSqPwtibBlXdzxGQYg=
 v0.5.0-rc1 h1:Abe2XAxQvxB0kOU7iZcOB3M7sbDPKDANo+LUartxD/8= -
 v0.5.0 h1:WZMajbGq20Tl/0YqV1H/w6hsNYBNxGLMfmGfZXuz11U= -
 v0.5.1 h1:Zcq7lwXhKlK0YvX1NFoe9/S6HmSnUOf2NtPLIaEWt6w= -
 v0.5.2 h1:WVfxD2Vz56jN3cNrn7quw8VhvlIJN2J0weJ8430Byvw= -
 v0.6.0-rc1 h1:+YmNvtO3xCtj1J4RTVSpLvBs8lF6zt8YwbfroVjOLus= -
 v0.6.0-rc2 h1:Br5NU59nHYN3cGnyMqXiGi6FgQ+vTnVApBu50kQOm+E= -
 v0.6.0 h1:64tEiKTViakF5Qr84DJBsiU4nn0RbzYyGDgzgM2ZdxM= -
 v0.7.0-alpha0 h1:bniZFfNgKpSCemyzkN96FgmOAE2QwqEtVUHzwNaMJyg= -
 v0.7.0-alpha1 h1:QIGrZ0975kLCXO8ajYIWnki5VWv63kLnZTErG90mla0= -
github.com/appc/spec
 v0.8.2 h1:S6m0fAPENQhZtqKj9pIUQOtiQotORSDCGPBXXd+uzEQ= h1:2F+EK25qCkHIzwA7HQjWIK7r2LOL1gQlou8mm2Fdif0=
 v0.8.3 h1:E7jmK2XvAfx/iZqoBktJhNN5pkj2PPeglEJR4I7D4mY= -
 v0.8.4 h1:GSpyz7DJkRyIiT8Hs1uVXh6cI+NktlKBK8ZC0bEWRRc= -
 v0.8.5 h1:OiSWM9BSw23A8jfqQgzzR1S8NzUVg2kBfKDBx5Vurzs= -
 v0.8.6 h1:b4weYOUBjwN4iGvIQNp/WHhK44Kwx39Y9i+6Jr+5kt4= -
 v0.8.7 h1:5W/+DzI8oifyDN8D0VZ1f5RAISxSwzP+JxbqEDIwNPg= -
 v0.8.8 h1:B/Mpno+onKj2194aU9EXZFH/7SL+9MZx3vVHeaf6MH0= -
 v0.8.9 h1:AkT6bEI+KxGuy1cpbWeKLvtzQ0dVg2D/f0woPE8MvJ4= -
 v0.8.10 h1:I9eyBoOeIZ/y6RQPTrn4ar+dmu44Na/ddQl7xjjboIU= -
 v0.8.11 h1:BFwMCTHSDwanDlAA3ONbsLllTw4pCW85kVm290dNrV4= -
github.com/armon/go-radix
 v1.0.0 h1:F4z6KzEeeQIMeLFa97iZU6vupzoecKdU5TX24SNppXI= h1:ufUuZ+zHj4x4TnLV4JWEpy2hxWSpsRywHrMgIH9cCH8=
github.com/asdine/storm
 v1.0.0 h1:1lzwUG5Rh4VToqwwetiednC9DP7lhxjxDQViZcMp0yI= h1:RarYDc9hq1UPLImuiXK3BIWPJLdIygvV3PsInK0FbVQ=
 v1.0.1 h1:95fBmWeTocU/faZg7hxvmAM+tuNxrZEu6Bm4IrWnOy0= -
 v1.1.0 h1:lwDLqMMPhokfYk8EuU1RRHTi54T68EI+QnCqK5t4TCM= -
 v2.0.0-rc.1+incompatible h1:TqYwZ9gXnjud1dF7DMHbtJepW20qFIILn8ACcD9DmTs= -
 v2.0.0+incompatible h1:iG+Anu3d190WehVdyXGZ+VTWYOYTn8+pYuNORlCwwic= -
 v2.0.1+incompatible h1:TmurTbZAH+/kAwVfYOPQSQSzr7CATXgqwVaGinGA49E= -
 v2.0.2+incompatible h1:k4ApXZdCe5zYf9865+0zuvGAyMsXu1hbSXV1l4ez1P8= -
 v2.1.0+incompatible h1:8kbvyrzhTtb7TiwGGjsNSKuCsKqJsyvJeZslh9W3YN0= -
 v2.1.1+incompatible h1:j/IqbSqHVmrU908a11QGf+2Iv7pr7NXiyDE+P35Bp80= -
 v2.1.2+incompatible h1:dczuIkyqwY2LrtXPz8ixMrU/OFgZp71kbKTHGrXYt/Q= -
github.com/astaxie/beego
 v1.8.0 h1:Rc5qRXMy5fpxq3FEi+4nmykYIMtANthRJ8hcoY+1VWM= h1:0R4++1tUqERR0WYFWdfkcrsyoVBCG4DgpDGokT3yb+U=
 v1.8.1 h1:nVKkVtLNuoqESIj08jU5ZZVTWt5zuMf3ZRZK/It3GyQ= -
 v1.8.2 h1:Xq+l4k5xlGtMC3obQPvIy5pEqkKZy8pTyp0PHWb0JDc= -
 v1.8.3 h1:6SwgDPBxYSs+E2cwIL7BzQn4nWgY8xwYE8wO6YpNa9k= -
 v1.9.0 h1:tPzS+D1oCLi+SEb/TLNRNYpCjaMVfAGoy9OTLwS5ul4= -
 v1.9.2 h1:Jw8glCLKrXd7BL65WjsYsfquxO+dF0TvhBhSOP19mN4= -
 v1.10.0 h1:s0OZ1iUO0rl8+lwWZfPK/0GhQi1tFUcIClTevyz48Pg= -
 v1.10.1 h1:M2ciUnyiZycuTpGEA+idJF0gX24h58EbPvGqjnO/DCg= -
 v1.11.0 h1:5Ke/j7NfQQJ9/sKDgZMQkhTHm18k5dApQwqkAJwPfMk= h1:mBKqEBdFFvQNfhHLZeQKH3BTDVSCbJs5zGoFOU97i5A=
 v1.11.1 h1:6DESefxW5oMcRLFRKi53/6exzup/IR6N4EzzS1n6CnQ= h1:i69hVzgauOPSw5qeyF4GVZhn7Od0yG5bbCGzmhbWxgQ=
github.com/atotto/clipboard
 v0.1.0 h1:pw3q0vqdkRc2qob0PLEZo3NFSQehzW2dgSdbMI75kEs= h1:ZY9tmq7sm5xIbd9bOK4onWV4S6X0u6GY7Vn0Yu86PYI=
 v0.1.1 h1:WSoEbAS70E5gw8FbiqFlp69MGsB6dUb4l+0AGGLiVGw= -
github.com/aws/aws-lambda-go
 v1.2.0 h1:2f0pbAKMNNhvOkjI9BCrwoeIiduSTlYpD0iKEN1neuQ= h1:zUsUQhAUjYzR8AuduJPCfhBuKWUaDbQiPOG+ouzmE1A=
 v1.3.0 h1:ZlUgCFoY5Eebf7mDuOvjCfTzSwBvYPa2gR9O4cAxXow= -
 v1.4.0 h1:utQd5PalMe5a7bLjfcf6dmXInYmyTCAZG+1iOO0P8o4= -
 v1.5.0 h1:WY+2xr0O7ycu4pKCRZrrtq09b7s0HxEKlaI8LdD/vmg= -
 v1.5.1 h1:82HrN4zjDClq9EVmU4mhQxdVu4dQMPBZyy88T5j+Xw4= -
 v1.6.0 h1:T+u/g79zPKw1oJM7xYhvpq7i4Sjc0iVsXZUaqRVVSOg= -
 v1.7.0 h1:g3Ad7aw27B2lhQLIuK7Aha+cWSaHr7ZNlngveHkhZyo= -
 v1.8.0 h1:YMCzi9FP7MNVVj9AkGpYyaqh/mvFOjhqiDtnNlWtKTg= -
 v1.8.1 h1:nHBpP6XC30bwF6qWKrw/BrK2A8i4GKmSZzajTBIJS4A= -
 v1.8.2 h1:wC8KcAG9HyVkFjbKQ9uhp87UGZutlPn9IJPq9fYM2BQ= -
github.com/aws/aws-sdk-go
 v1.16.30 h1:8QLugp2+gbixFN85sGSR97qvaXsjTOVUrA2bbsLCDOA= h1:KmX6BPdI08NWTb3/sm4ZGu5ShLoqVDhKgpiN924inxo=
 v1.16.31 h1:bE4FW2uulhXiAaF4Guw0OzX9gBZ4iWvXWe6VT8Jxr28= -
 v1.16.32 h1:/grHp+bt3OAVWkdCQv2YtXkWuu58SuTlH1U8tp25n1c= -
 v1.16.33 h1:jXrsqeNbpLkM4TrnZbtr+4k4x7frwcLP3DiWMa7NOtE= -
 v1.16.34 h1:kZj0biNt+YfvqC11/NtMGqB6YpHXd9bVEzmcTT4CmJg= -
 v1.16.35 h1:qz1h7uxswkVaE6kJPoPWwt3F76HlCLrg/UyDJq3cavc= -
 v1.16.36 h1:POeH34ZME++pr7GBGh+ZO6Y5kOwSMQpqp5BGUgooJ6k= -
 v1.17.0 h1:+pbWEdKxH1qlLb07as1+auEVvx+IxkaDzQLwMzbK1tI= -
 v1.17.1 h1:RoOo57SetcPFGQ6vesLfWIpfnsbpEiuwiHq6aCvjrZw= -
 v1.17.2 h1:92HvIn2MROLHcidibvnzy7D0iHCygmonkNQKACbAvuA= -
github.com/aymerick/raymond
 v1.0.0 h1:jgij7LYInsgEGnvnlPILiGn8xgyfyfQU+iKZK0VAwNU= h1:osfaiScAUVup+UC9Nfq76eWqDhXlp+4UYaA8uhTBO6g=
 v1.1.0 h1:phuNN2s67eI/HtO8CrvqFcdR2JP+BtkGJZ9n692Hr2Y= -
 v2.0.0+incompatible h1:oDFxJCwQshTjA7YcdZJJMthIG1f3zDCVrLiCi9p11EI= -
 v2.0.1+incompatible h1:ZhYb+Bw5DNBMAl/UpvbxXP7pALGiMzCAE56QwHPqjjk= -
 v2.0.2+incompatible h1:VEp3GpgdAnv9B2GFyTvqgcKvY+mfKMjPOA3SbKLtnU0= -
github.com/beevik/etree
 v1.0.0 h1:gQ0/0GdWwIZONSQVL/btX2rZ/OwMSV7twGyq42D+KUg= h1:r8Aw8JqVegEf0w2fDnATrX9VpkMcyFeM0FhwO62wh+A=
 v1.0.1 h1:lWzdj5v/Pj1X360EV7bUudox5SRipy4qZLjY0rhb0ck= -
 v1.1.0 h1:T0xke/WvNtMoCqgzPhkX2r4rjY3GDZFi+FjpRZY2Jbs= -
github.com/bgentry/speakeasy
 v0.1.0 h1:ByYyxL9InA1OWqxJqqp2A5pYHUrCiAL6K3J+LKSsQkY= h1:+zsyZBPWlz7T6j88CTgSN5bM796AkVf0kBD4zp0CCIs=
github.com/bitly/go-nsq
 v0.3.6 h1:mZGagnVsEMN02ciyD64Rh6+tThqKAZPq4bPq7kspgrw= h1:uRcXTyAr/ggVrYpmvkE4jkJNSWov99Gg4fwhhypc/P8=
 v0.3.7 h1:p7Ul2pL+elqkWC4NRbU8A/KoL3b6aX33fdX0NMNXtHE= -
 v1.0.0 h1:KrtGIL9MmHO2ae6OVA4MwydLk+moX9LZXdbBRxBivFk= -
 v1.0.1 h1:EOOQyoNbBuOJ1dpoYWmAqL+Xdu+8r+KjwrqJ3RK6jFs= -
 v1.0.2 h1:kUNcpLB49+sgN+9NoVNXdkeyDnzGuw4b/dPstbIGdjo= -
 v1.0.3 h1:HMVQuDxXhbd5w3ONX68VKsByESU5v7qac4U5koZRHqc= -
 v1.0.4 h1:mZgOcsNi1JC/TP8NmuiQ4BSmmjnBIcSYAZrFejgPRHg= -
 v1.0.5 h1:AoRjCf5NTjRC0MWdTEG5zYJlNuCZhhFrtrgLXfsK7zI= -
 v1.0.6 h1:RVVHxsteQEZ9iC9v5/RUWrcehXDJSmCVYtzsQ2jtr9I= -
 v1.0.7 h1:o6nk7C1LyG9wAEsD7AfPExMpLwfbc0oYwBhzQxqefQM= -
github.com/bitly/go-simplejson
 v0.4.1 h1:TGpmk+yOfSuvI75kj5haORCr+WL+I5zqTbHnZRvxSS4= h1:cXHtHw4XUPsvGaxgjIAn8PhEWG9NfngEKAMDJEczWVA=
 v0.4.2 h1:FMBg8cDyjr0EDQUGi4Qcy4vyqEUb/yhoqpVyWbAUgjQ= -
 v0.4.3 h1:F3lPub5ZygB4mLWK0UsYd5T7JI5fSmTyGdkJEGFegno= -
 v0.5.0 h1:6IH+V8/tVMab511d5bn4M7EwGXZf9Hj6i2xSwkNEM+Y= -
github.com/bitly/nsq
 v0.3.2 h1:Z1ODLjmJgdX53aS1/2us6av+fhqLB0HzhLNZBnD1wgE= h1:iscvdoiQoaXC4V7XCMmZOxr3PZGYZvMcRsQf8uTt/PM=
 v0.3.3 h1:AdzY1brGnQGBQa1LrAthBjFaLEl52MfT5t2XqGRlurc= -
 v0.3.4 h1:sielleZKJAqfm/mUWx1/yg0lhdVx7SB1yojmPL34QVM= -
 v0.3.5 h1:A6rC4frCmc0uEcjMpHnPhV/FKu+DX7MOr8o1Oe5scHA= -
 v0.3.6 h1:NPJzvj2y0asyv91epTzhZpiaJstvdB1QvvCMF/zQTz4= -
 v0.3.7 h1:XiBOFohN5AnQUM4xjd2yvLFuJn0xaHcitLtoRds7svs= -
 v0.3.8 h1:rqEmrDKjLDP9zWzyV62Ea1FNVUHgfAUeI9RhR8zm5xk= -
 v1.0.0-compat h1:MxyHntD0McEkAEWrLa8C+E345HG2DdVOuH9tq8+WXXs= -
 v1.1.0-rc1 h1:0Acse3L7stpf/MnFqLtn9Aw9VbxCm/+2mYu5+eSxy1s= -
 v1.1.0 h1:oEQihaX5y1E4UE8nWKXDzSq0PQVEyj0SMTZPV3jxOmo= -
github.com/blang/semver
 v2.1.0+incompatible h1:TtbN5kU/oDXfHg4iuCEQ+vUTdr8ED02yUWG0f5VPVs8= h1:kRBLl5iJ+tD4TcOOxsy/0fnwebNt5EWlYSAyrTnjyyk=
 v2.2.0+incompatible h1:DIb+hEi/XKX6t9Cvy5+oSlANqmc0eenMxbNBvLqpV2A= -
 v3.0.0+incompatible h1:N7yr3lQBIWoA+v1DXf5skCrTH62zKbkHmWDy+p0pYAY= -
 v3.0.1+incompatible h1:HSK4fJAkncdAqOFSWJBP6JslojPJ4G0Jn1uKwxzWJ1o= -
 v3.1.0+incompatible h1:7hqmJYuaEK3qwVjWubYiht3j93YI0WQBuysxHIfUriU= -
 v3.2.0+incompatible h1:HfFKH+psVkEzPzfJC0gUWrirRa4ERIzwAiGBNQNWMO8= -
 v3.3.0+incompatible h1:BRtK3PUrMsESEGZwVNZa3sYPGIoNRla1Uhy+oHXiXrE= -
 v3.4.0+incompatible h1:9A/N25tshFofYIu1794iWJshVXtvliOyciudY5/cfpo= -
 v3.5.0+incompatible h1:CGxCgetQ64DKk7rdZ++Vfnb1+ogGNnB17OJKJXD2Cfs= -
 v3.5.1+incompatible h1:cQNTCjp13qL8KC3Nbxr/y2Bqb63oX6wdnnjpJbkM4JQ= -
github.com/blevesearch/bleve
 v0.1.0 h1:BfNXayKYKM9fzUp5YL6m8qIqv2DstRyRj2sk8PXalaQ= h1:Y2lmIkzV6mcNfAnAdOd+ZxHkHchhBfU/xroGIp61wfw=
 v0.2.0 h1:vJUXB7ZTkrcu0uBx4duWimnEZ7VDdDqBow1HX1lRkDM= -
 v0.3.0 h1:5qRvr9qP/5IMyJBh64SVFmnbc0cbSfSRQ5on3GYiEFk= -
 v0.4.0 h1:dErdGEYlXMMVJCDJAf/Lqu1DAKC0RsKgty+WJrxFHyc= -
 v0.5.0 h1:gak5LhWxIJjGz3wTisxKLcXfxcx+0l7Sk2gEGsnTyu4= -
 v0.6.0 h1:lPg4qruuvBQWQpAPmHgvcIs1ibsbWC/dbca6e1fc8Mk= -
 v0.7.0 h1:znyZ3zjsh2Scr60vszs7rbF29TU6i1q9bfnZf1vh0Ac= -
github.com/boltdb/bolt
 v1.1.0 h1:o7Lse6JSBiorjJqUfn/SHYKJ4tKfmzVmO3tgnkEizFI= h1:clJnj/oiGkjum5o1McbSZDSLxVThjynRyGBgiAx27Ps=
 v1.2.0 h1:4FPRe2N+UvvrhrU7U+A60z9qaK75Q0wDmxjxZrMntTQ= -
 v1.2.1 h1:dqty3m0dX4pwSiqpaNlLlaVw33YmC0NWCVpo4PJmxyA= -
 v1.3.0 h1:am1Tz34FiDO8OP+gvSpAeYb6Iy1lME5KHxZoFXbfbLs= -
 v1.3.1 h1:JQmyP4ZBrce+ZQu0dY660FMfatumYDLun9hBCUVIkF4= -
github.com/boombuler/barcode
 v1.0.0 h1:s1TvRnXwL2xJRaccrdcBQMZxq6X7DvsMogtmJeHDdrc= h1:paBWMcWSl3LHKBqUq+rly7CNSldXjb2rDl3JlRe0mD8=
github.com/bsm/sarama-cluster
 v2.1.5+incompatible h1:qBhKDZ3LO4u1jqs53HAWJZ/P6AFf/DbaGupjETps0os= h1:r7ao+4tTNXvWm+VRpRJchr2kQhqxgmAp2iEX5W96gMM=
 v2.1.6+incompatible h1:9taGDgI818Tx4ynNFIOr6vfu3ZmC3dyWD1zhx8e46kk= -
 v2.1.7+incompatible h1:VJcuQ/oU29WnXwj3vk4gM+O/46qqg4Yd0Xxh5yL0p9Y= -
 v2.1.8+incompatible h1:/Qfd5p1pXhSXzafAayTPMa/d1fMzZgZbeFqszsAe5Io= -
 v2.1.9+incompatible h1:EdayzygiBrKtrj2exJtHHpWrN96nKvnLizN7aNWlpcc= -
 v2.1.10+incompatible h1:+8EWPWOItwlMNEi5bqy7HgO8x5DO97CEyS3AWiw9MiQ= -
 v2.1.11+incompatible h1:eBCnGwJcydiX4g9yabG/TfKvOMb3UeuAwBVuTE+2hN8= -
 v2.1.12+incompatible h1:7NakmL2HnNvCCy685HlJbhYDU390nanFpwoonmhDJlw= -
 v2.1.13+incompatible h1:bqU3gMJbWZVxLZ9PGWVKP05yOmFXUlfw61RBwuE3PYU= -
 v2.1.15+incompatible h1:RkV6WiNRnqEEbp81druK8zYhmnIgdOjqSVi0+9Cnl2A= -
github.com/btcsuite/goleveldb
 v1.0.0 h1:Tvd0BfvqX9o823q1j2UZ/epQo09eJh6dTcRp79ilIN4= h1:QiK9vBlgftBg6rWQIj6wFzbPfRjiykIEhBH4obrXJ/I=
github.com/bugsnag/bugsnag-go
 v1.0.5 h1:NIoY2u+am1/GRgUZa+ata8UUrRBuCK4pLq0/lcvMF7M= h1:2oa8nejYd4cQ/b0hMIopN0lCRxU0bueqREvZLWFrtK8=
 v1.1.0 h1:0jF7rytU+6wd0X26evrxXzrRQF2/wYndVYcTsmKyeYw= -
 v1.1.1 h1:FH9mZaDrqDMX9FydxWZXM+ypVc5G4dKGxk7nxhpnt+g= -
 v1.2.0 h1:zGZiV73MHmlrWR9dj5yMj0FaZGMpaQC9sMrB/0P9DwQ= -
 v1.2.1 h1:d0SjXdWdFOK224Bbhfna6xsgClo1hKDtRsekbgHBlL8= -
 v1.2.2 h1:u0nv8EHnaZ8Eenc+d0nfGvNO/YIjNTlHXB2dKr9S75E= -
 v1.3.0 h1:vIHdYCVDBtzRcj5wvxljvbnFduWpYGPt3esulAsX/Vk= -
 v1.3.1 h1:NFDoCNG9B2lx1+zQqJgayvO1prIdXwMmHLZn8zMT6ZI= -
 v1.3.2 h1:8bcRylldQKQiAx9/KPu9+1iLZwgK1eN1Ib3SROSXfIY= -
 v1.4.0 h1:CLCt5wO6/P0GelBEMRrlF52XveQMnnXHoCoxGZ+8a5g= -
github.com/bwmarrin/discordgo
 v0.10.0 h1:sDDvLPcuCVHA+KzbwO4WKQ2vH8lKKxmYGoJCIXqVkDg= h1:5NIvFv5Z7HddYuXbuQegZ684DleQaCFqChP2iuBivJ8=
 v0.11.0 h1:hi1bjS/WuHgWxdkZO0WR2WVsjLXnJVFv0iM69RZkW4Y= -
 v0.12.0 h1:QdNqxQyACMk+nPRAfKSYoMYyRUuWyBU8MXU1BLUeI+8= -
 v0.12.1 h1:2rbWXqbzc1XUUnbmBLp3lor2Uhe/xa1igR5YnvWnCH0= -
 v0.13.0 h1:Dz2Oa+uMm9kFKMPWZ/LEtW2+20SkNa0fA5/fi0c9Z2w= -
 v0.15.0 h1:WXViLKrvxaMG5K1X03qgO03mSCqemiA1xFEv1EezurM= -
 v0.16.0 h1:/HhaLf7VXwJe/zcN+i/tKIbhKa1Y9Xy0uFXHyiDm7TU= -
 v0.17.0 h1:VUZc9ppGi0j9xKsRBYctJqFEFl8u0FSMzDBObw705cM= -
 v0.18.0 h1:XopVQXCIFy7Cr2eT7NcYcm4k0l2PYX+AP5RUbIWX2/8= -
 v0.19.0 h1:kMED/DB0NR1QhRcalb85w0Cu3Ep2OrGAqZH1R5awQiY= h1:O9S4p+ofTFwB02em7jkpkV8M3R0/PUVOwN61zSZ0r4Q=
github.com/caarlos0/env
 v3.1.1+incompatible h1:DfTWQ5cr8n51wmIGYJKAHeWJx5dryHk9N3fgRhItS4A= h1:tdCsowwCzMLdkqRYDlHpZCp2UooDD3MspDBjZ2AD02Y=
 v3.2.0+incompatible h1:47SrI1EMf4OwTL8DaHmjVU/A/hmoUeu8L9FQQRXknsQ= -
 v3.3.0+incompatible h1:jCfY0ilpzC2FFViyZyDKCxKybDESTwaR+ebh8zm6AOE= -
 v3.4.0+incompatible h1:FRwBdvENjLHZoUbFnULnFss9wKtcapdaM35DfxiTjeM= -
 v3.5.0+incompatible h1:Yy0UN8o9Wtr/jGHZDpCBLpNrzcFLLM2yixi/rBrKyJs= -
github.com/cactus/go-statsd-client
 v1.0.1 h1:R58kYEe8CS4Dg7zNgGi8J0nFhYIGaBvIC7SHJ1l08is= h1:cMRcwZDklk7hXp+Law83urTHUiHMzCev/r4JMYr/zU0=
 v2.0.0+incompatible h1:sxAVNmd64vydbUsHgVIzR4zmXgnJezfZJUORu1/cgu4= -
 v2.0.1+incompatible h1:i67s5+3NqhEdmVtGIUZJdQotCMgYUQ//5nR1YxdIKp4= -
 v2.0.2+incompatible h1:fq6u7pZGQh45DMCKGzLPL4VUY6YOpQbySJE6yLsIedk= -
 v3.0.0+incompatible h1:RkB08vazVUSx0MRnt824Ohv7GYXffZ16piKf3m3W/sI= -
 v3.0.1+incompatible h1:Fk6etBCheGhbrRmfHuaetxZ6H9/Mp2xl4D+Dcxo19zo= -
 v3.0.2+incompatible h1:vjpspf/8TEt85VK6/pVEEP1Blxhl7rE1puzib/gh6Zs= -
 v3.0.3+incompatible h1:Kipzu5qAaQ1Sipag8JNCeyWayxBufo19Xont/KXHTfE= -
 v3.1.0+incompatible h1:jtloShmaP/MkAW68aaWwQZrzlOUXVLudFmBQsskTs7A= -
 v3.1.1+incompatible h1:p97okCU2aaeSxQ6KzMdGEwQkiGBMys71/J0XWoirbJY= -
github.com/casbin/casbin
 v1.0.0 h1:DyIqMSuwLkClTBDndMogDOrT8RT60rAltOVg6EHPFP4= h1:c67qKN6Oum3UF5Q1+BByfFxkwKvhwW57ITjqwtzR1KE=
 v1.1.0 h1:jXVs+Zf2QdV1fLP0Yu0N/aS/OhU19wMAyl1U260pF3w= -
 v1.2.0 h1:NNSzEev1OtggZDIVz68PnCC+AaPkG+obxGLS184L/Po= -
 v1.3.0 h1:EQmPGgOo8/veRZbg+w2aspCFHducIoq6a4wEYRcD3RM= -
 v1.4.0 h1:TCykTIM1VrxrEsglLtp4cbDHF0GwPU/pjMKxRpRmnJQ= -
 v1.5.0 h1:mu575Bh7CW6JrjAjnQz0sTtHJvhpf3Gm5r2+tsWn6AY= -
 v1.6.0 h1:uIhuV5I0ilXGUm3y+xJ8nG7VOnYDeZZQiNsFOTF2QmI= -
 v1.7.0 h1:PuzlE8w0JBg/DhIqnkF1Dewf3z+qmUZMVN07PonvVUQ= -
 v1.8.0 h1:eEDIzfiSg6aR5lqeQQ+YUhVLccsxykq1zcpWFOI4Kxo= h1:z8uPsfBJGUsnkagrt3G8QvjgTKFMBJ32UP8HpZllfog=
 v1.8.1 h1:BVvL6H0nc+1y68nwIe8ZxwMIOEVUgg9y00yeD3GTDCc= -
github.com/cayleygraph/cayley
 v0.5.0 h1:0vUVlZXasya23TrU7EYkMuC/MKYYOISf67r1VbSWCkQ= h1:8bQ/gpnJrp0qT1pSNE6eLN/f57dfAIFaVieI4vngp/A=
 v0.6.0 h1:/ymlPTlkFE4FTLUl8uXNZM65JY+a3OcB/WB2XIaEH2Q= -
 v0.6.1 h1:1FDMG/UfXk25dqnckxCACl9YecKqGQGmsMBMAogyZ7g= -
 v0.7.0 h1:H8S4MEYZkO9IWEXfzpTJ8P3swqn9DbGlkiIKoF616wE= -
 v0.7.1 h1:dQj+0jIlGV/cxIiWjO0ZmEnJS9PH35cFtTE3v6CAWI4= -
 v0.7.2 h1:65zvSjK6VNGF92z5QhIcXLSo4v7q0IOc4V6hkDh77gQ= -
 v0.7.3 h1:k7EdGsh+eRkL16stnozfJXCQOPy9ZOMKJ3B+bpXpOQI= -
 v0.7.4-2 h1:UZ6aPfbNWCNx3kYXCqpxLWSuwJ5zraT+E9Bn8HdV6hc= -
 v0.7.4 h1:ND1LuXaAk53ARjR9GxYHpO8NcTqgCvuckCTZHVbTYRw= -
 v0.7.5 h1:eUDWKF/Xg/N95ihuAdUKVPQ+pgaOvuvQ9h0XE3ZAPic= -
github.com/cenk/backoff
 v1.0.0 h1:G5x+fd8E4lParOjah6AuRY161LFjd2hAWyx6jI22b9s= h1:7FtoeaSnHoZnmZzz47cM35Y9nSW7tNyaidugnHTaFDE=
 v1.1.0 h1:Smh2SqufUB51+RcyQ7/2tkPCGazvVSrqT96C80dFtr0= -
 v2.0.0+incompatible h1:7vXVw3g7XE+Vnj0A9TmFGtMeP4oZQ5ZzpPvKhLFa80E= -
 v2.1.0+incompatible h1:WZ2V3Qku5F7D7FC7l7/M9LV2i7fmIpVtpMUr3GiGU7k= -
 v2.1.1+incompatible h1:gaShhlJc32b7ht9cwld/ti0z7tJOf69oUEA8jJNYV48= -
github.com/cenkalti/backoff
 v1.0.0 h1:2XeuDgvPv/6QDyzIuxb6n36ADVocyqTLlOSpYBGYtvM= h1:90ReRw6GdpyfrHakVjL/QHaoyV4aDUVVkXQJJJ3NXXM=
 v1.1.0 h1:QnvVp8ikKCDWOsFheytRCoYWYPO/ObCTBGxT19Hc+yE= -
 v2.0.0+incompatible h1:5IIPUHhlnUZbcHQsQou5k1Tn58nJkeJL9U+ig5CHJbY= -
 v2.1.0+incompatible h1:FIRvWBZrzS4YC7NT5cOuZjexzFvIr+Dbi6aD1cZaNBk= -
 v2.1.1+incompatible h1:tKJnvO2kl0zmb/jA5UKAt4VoEVw1qxKWjE/Bpp46npY= -
github.com/cheggaaa/pb
 v1.0.25 h1:tFpebHTkI7QZx1q1rWGOKhbunhZ3fMaxTvHDWn1bH/4= h1:pQciLPpbU0oxA0h+VJYYLxO+XeDQb5pZijXscXHm81s=
 v1.0.26 h1:cxVZXxXCTNW7yYwnrTAhJ42LcWrLjp676j+y1AmmLKA= -
 v1.0.27 h1:wIkZHkNfC7R6GI5w7l/PdAdzXzlrbcI3p8OAlnkTsnc= -
 v2.0.0+incompatible h1:Y3hEZw6ljXSjE0YQ3LueJMaedMkzrnfqV4fL6yGAzO0= -
 v2.0.1+incompatible h1:Mud6xcHpxgb7qm2tWQFyQnIe6nTfBtDxFiRQ2IHSahM= -
 v2.0.2+incompatible h1:09likKXoFogpRXyRLk+Goj1i6nACYJQn3T7pN3T0wcw= -
 v2.0.3+incompatible h1:re76+eCsiyHF+4mmVDJVHYq3UMiDAbIASryJyPEnvkI= -
 v2.0.4+incompatible h1:WAfRRtSHyWSGAHdcgsfeli9rh65YLinhaOzwuwg5LoQ= -
 v2.0.5+incompatible h1:zJxULLIJRvO2VkqeRT1mX7e3oxuPnaFz9AXpEPNJ04s= -
 v2.0.6+incompatible h1:sutSx+mRaNbeJUMCAtyqNWU/tQ0B/xBm+hyb1JQmQYs= -
github.com/clbanning/mxj
 v1.6.1 h1:o+OW3m8R6L1hsUHBKGNmYChs8GDIFSPFtmeXT7fKW5c= h1:BVjHeAH+rl9rs6f+QIpeRl0tfu10SXn1pUSa5PVGJng=
 v1.6.2 h1:BofqUYikgGcG/5OURYR5IF2LM4dPGZ93swoA80uxEBQ= -
 v1.8.1 h1:5LCSILVRHt5NJKN04xonqopYq3l/vQm6oqBeC2oXYGc= -
 v1.8.2 h1:KBWvavOh0B3laROMyum5QODKHVA8RkTZGyO1SrgJkRI= -
 v1.8.3 h1:2r/KCJi52w2MRz+K+UMa/1d7DdCjnLqYJfnbr7dYNWI= -
 v1.8.4 h1:HuhwZtbyvyOw+3Z1AowPkU87JkJUSv751ELWaiTpj8I= -
github.com/cloudfoundry-community/go-cfenv
 v1.14.0 h1:rgK31f+7xrYkdfBGNDRqmXifmQgp4xKNu3uiFAnATOM= h1:2UgWvQTRXUuIZ/x3KnW6fk6CgPBhcV4UQb/UGIrUyyI=
 v1.15.0 h1:KbnZgqqLp4O77PpZxkO/N7qY2dyWwzXLUHkGFYe5Heg= -
 v1.16.0 h1:GRWTi7YruQRzHK4Xfe3iGnmkVQCe1ECirsv454NXUoo= -
 v1.17.0 h1:qfxEfn8qKkaHY3ZEk/Y2noY79HBASvNgmtHK9x4+6GY= -
github.com/cloudfoundry/bosh-agent
 v2.189.0+incompatible h1:Yh36in52BqIAqA5lc6pmHne6mA0ry1Bdc021WVp1WrQ= h1:7UvVn5vc/d6icLrBx6GhBlpSMwe2+x1C2A7x4TbPhiU=
 v2.190.0+incompatible h1:SUns8ndwpkcKqyYcfoI9xDBzL/qPaqKOk5FCHZRh3no= -
 v2.191.0+incompatible h1:IDnTYBeFGR+VWbsRmYiGtgMbQoQCceDtlWomOHjZVic= -
 v2.192.0+incompatible h1:Fxwt9m7MEH8g2vJ1zW8Hp44v5MyUbIfI0LtbBEcR0/U= -
 v2.193.0+incompatible h1:ExQHIDdP+hx6ZZtVXXvFzJrfgP3EJmLdJqTSKGl/Edc= -
 v2.193.1+incompatible h1:N+c+2E52Rp89ZFoL7maQssJx6bsKLASx0VM+nXnJ2FE= -
 v2.194.0+incompatible h1:18EIhDBQgwAM8uLGhK88HdQsIQRpQOQJDhiv7my6tSI= -
 v2.195.0+incompatible h1:9NSLZq5VrflA18SGcPpdKTca6VvzzhBT4K1x7dJeaXQ= -
 v2.196.0+incompatible h1:mr1/+uLzVG24S2Pnz79WbKYUf6rHxInUsAa+bnqf134= -
 v2.197.0+incompatible h1:kquzwyJL1Vd1K5L6+1QL87DmAzqxB+6KG4VyF0iDd8c= -
github.com/cloudfoundry/cli
 v6.36.2+incompatible h1:m8zsB/Y4TFagyzwaI5cOnf/obYyCzQixrLwB9GjTnZo= h1:uUVSLzSuwWNhis5+tY5XRUp66kLbHhBktg8b3ZfcJHI=
 v6.37.0+incompatible h1:+qmgK5qS6lp0H9p/3SuyaBGwb+OW0YcM+5iCdhfUA2Y= -
 v6.38.0+incompatible h1:K/T6wDlaQwafLTfXeR/xpo2Wy1KkYBgU4Me8nR2xPH8= -
 v6.39.0+incompatible h1:UepfMeLAD2BvwZ1EfP9FhOuqQKbH5AjQwwIZaZwQs04= -
 v6.39.1+incompatible h1:jqoGunW7txeJN3rRms/cO3QSJsQAJgfauP1nmhGZE70= -
 v6.40.0+incompatible h1:AlWpueUXdvQmD9DO6Evt5X5O1hNmJXfwRUZD6k4yWqQ= -
 v6.40.1+incompatible h1:0SOgAFaxL3xX7GJ+ZvsY2ivAsqQ1tcY1aMoI2L/D3P4= -
 v6.41.0+incompatible h1:q3TUA3szij8K4opC2UXrqmvXBdDytPy/tYmtXWYVfy4= -
 v6.42.0+incompatible h1:wccca2Nl23Ulk/fzZmrNNu/ETYxmVlIPvwdWJK1SkGg= -
 v6.43.0+incompatible h1:ek6RenuwsD2GxOtkBY4206Oqh45OEUinMsXQ02cW/iE= -
github.com/cloudfoundry/dropsonde
 v1.0.0 h1:9MT6WFmhU96fQjhTiglx4b1X3ObNjk/Sze7KPntNitE= h1:6zwvrWK5TpxBVYi1cdkE5WDsIO8E0n7qAJg3wR9B67c=
github.com/cloudfoundry/gosigar
 v1.1.0 h1:V/dVCzhKOdIU3WRB5inQU20s4yIgL9Dxx/Mhi0SF8eM= h1:3qLfc2GlfmwOx2+ZDaRGH3Y9fwQ0sQeaAleo2GV5pH0=
github.com/cloudfoundry/noaa
 v2.0.0+incompatible h1:+CeKb9WVomnhrFSQAvYuKgKjflElt6//aDlpIyAnNVk= h1:5LmacnptvxzrTvMfL9+EJhgkUfIgcwI61BVSTh47ECo=
 v2.1.0+incompatible h1:hr6VnM5VlYRN3YD+NmAedQLW8686sUMknOSe0mFS2vo= -
github.com/cockroachdb/cockroach
 v2.1.0+incompatible h1:snoi1AnyS2ol1MH5Kap7ELcw1JpyWUygo2kORSyWWd0= h1:xeT/CQ0qZHangbYbWShlCGAx31aV4AjGswDUjhKS6HQ=
 v2.1.1+incompatible h1:zqb5A7qOXZ6apceiziiog51YNmxTHz/NoZOZl5EopMY= -
 v2.1.2+incompatible h1:/HKe0BNF8zcdMXbYcg+vFlIEVw2WfJ0U+oWkQ79WORM= -
 v2.1.3+incompatible h1:niMEggd4J7fImJIHAlMC0q0nnylF+SMZ2K1mqdoY8jQ= -
 v2.1.4+incompatible h1:4al/4oL8KCbd28jly4RgFZnUakYrEDIByyeA/RrqXhQ= -
 v2.1.5+incompatible h1:6UVgk4e1koeJduE1LaxCJzVHGo6qtGO4FyeFdD0S/KU= -
 v2.2.0-alpha.20181119+incompatible h1:yrctfZSVelQcXI2KVEFgBn+Xuwu/8uggGFqvuev7fPg= -
 v2.2.0-alpha.20181217+incompatible h1:qmakWkQz9zaD3cl3q+EmsdwzjfjWu5TjJLyDAdMbUmM= -
 v2.2.0-alpha.20190114+incompatible h1:nzgynLMrd812xj166mUqG2e8RjUhC3XiHLigJ1AIK0Q= -
 v2.2.0-alpha.20190211+incompatible h1:Kk5g/ndezayIj7F3LQCFeTNe+yGGJsj3TWsybhytgsA= -
github.com/codegangsta/cli
 v1.15.0 h1:XQ8cvKGFCHeHZsqkyyjsKG2i4v7xoEuEbLqiIFQZ5RY= h1:/qJNoX69yVSKu5o4jLyXAENLRyk1uhi7zkbQ3slBdOA=
 v1.16.0 h1:A4v+152JTrkyHhedr4bIuZMVMS3ydNfvpXCq6Uxj4W0= -
 v1.16.1 h1:aZvyIKV0pjB812U6tD71U7m0bbTxDBHggFNagvTfyhE= -
 v1.17.0 h1:K1LmHEkqORKbkRnb0YacGC+dR2+xeN6v+f/pFniQwJU= -
 v1.17.1 h1:aBmEKUaSVLD8WP1d4E9JSiR0pbOc/fNQB4PI7vVdFyY= -
 v1.18.0 h1:K+szYvi2B28coQzhPZGac9o/JS0QZkvYAjjVi0+O2cA= -
 v1.18.1 h1:VohEUscvOrgFKLWz3iq/OkBayUY3yG/wVUYQtj5RPes= -
 v1.19.0 h1:UJrnqaJr2Qy3WHi9T2jZmubjk4JudCdTcrnkAm5IDzk= -
 v1.19.1 h1:+wkU9+nidApJ051CVhVGnj5li64qOfLPz7eZMn2DPXw= -
 v1.20.0 h1:iX1FXEgwzd5+XN6wk5cVHOGQj6Q3Dcp20lUeS4lHNTw= -
github.com/codegangsta/negroni
 v0.1.0 h1:VapIFHO0LpzRe15N2GqEVsAy5EaUs6HpzE5Wag450+4= h1:v0y3T5G7Y1UlFfyxFn/QLRU4a2EuNau2iZY63YTKWo0=
 v0.2.0 h1:MqNkcR831ZSHwZmP9IdlZWGMxc1T8briBhBW649DkXs= -
 v0.3.0 h1:ByBtJaE0u71x6Ebli7lm95c8oCkrmF88+s5qB2o6j8I= -
 v1.0.0 h1:+aYywywx4bnKXWvoWtRfJ91vC59NbEhEY03sZjQhbVY= -
github.com/colinmarc/hdfs
 v0.1.4 h1:7inMUiDV79eSDV2NNp+/9nq5w3vP6E197l7CiYlaNLQ= h1:0DumPviB681UcSuJErAbDIOx6SIaJWj463TymfZG02I=
 v1.0.0 h1:WF4obl7LEXYuVnv9PKI6sKpH9fEmyg5yGByOlo25/vE= -
 v1.0.1 h1:L7hzAmmaf4m0WYagoW+HIi+kN3D439ihmd7kEpN3aos= -
 v1.0.2 h1:KH25/yu1lNPE/KU3IZDnB2IZ23hPuQrj76ZBz+JQ4Fk= -
 v1.0.3 h1:nC283+++Nyrl2RznLQAVprB7T+myovSgqq64kosP0Ek= -
 v1.0.4 h1:ig284slxQJI6PfRKpytNMFSmC4MFW8h7YlcowDXHzsM= -
 v1.1.0 h1:eUpeUxFxahhQ4kGhagKuHzwURmMwYOJ7WtCAMpfblzk= -
 v1.1.1 h1:/CIwLFhGUOiC9q3E4/H2RPHiS7fD9sU776puryFlLq0= -
 v1.1.2 h1:mLSI9XCNEbhylIuNRtEtN92lLJ4Qvdyoj0a6FsgxX8E= -
 v1.1.3 h1:662salalXLFmp+ctD+x0aG+xOg62lnVnOJHksXYpFBw= -
github.com/containerd/containerd
 v1.2.0-beta.2 h1:IYGcR47Wxj8k8+jIF7882PaG4VcIK8cBtXHKULsw3Bs= h1:bC6axHOhabU15QhwfG7w5PipXdVtMXFTttgp+kVtyUA=
 v1.2.0-rc.0 h1:OafxxFaCdxGzzaqmsJIdu1QLDI4OnG1natz3h4Wp0Ao= -
 v1.2.0-rc.1 h1:gNCIPosHSxnfW3JGo007/hdHLp2KRBYBC7Se6nzJwlc= -
 v1.2.0-rc.2 h1:h0zfrUgy5Xb9zjwqlCVOtmUDswFebXC9TBj1Ynx2Hy8= -
 v1.2.0 h1:0NP+uCCcSf7IHOEw/WE1vgcKh6DKIlTqqfh+dtoxDhw= -
 v1.2.1-rc.0 h1:YDXzJFhpqcsrXEQvk3VXBh+CFDff0lbYUbWSbt+9tdc= -
 v1.2.1 h1:rG4/dK9V2qa5a9ly/E3CtG6/FBXfmSkDo8An3ea2Yt8= -
 v1.2.2 h1:N3tAHxrX+byqfAsENdDWLSMtFD4thUxK7kFElUl+8z8= -
 v1.2.3 h1:6TvLXAk7vuAF72J3p6Fcw6c7Z8CDZFbZrHl+XIovyqk= -
 v1.2.4 h1:qN8LCvw+KA5wVCOnHspD/n2K9cJ34+YOs05qBBWhHiw= -
github.com/containernetworking/cni
 v0.4.0 h1:oWucloJfoPBFrFUGJ1JB69DI5Y6gL7GWIBEU69fzSvo= h1:LGwApLUm2FpoOfxTDEeq8T9ipbpZ61X79hmU3w8FmsY=
 v0.5.0-rc1 h1:70TFSlNcsSxNp0i2K6NeXcsgx9/x3juCXRDBNBIrNCw= -
 v0.5.0 h1:nyWKaB+/FIk5B4t1msH8yoZCZU5Z+H1bl9nOoC7kU4g= -
 v0.5.1 h1:0L9JEX+A/WfDG1kp5PXrdD8uWtlJlNJuLnhizJzrDWo= -
 v0.5.2 h1:/nFPNGJQu4yiNvXxH31qL05FIyGG5Y/p0YxuUutl+PE= -
 v0.6.0-rc1 h1:BQ2TcgoQbdbk5SLaUTY+N282hMhoI89QZd+9CIhvA84= -
 v0.6.0-rc2 h1:re+FKi1FMnMMy3nY3iYuMdtyJR2kk2KTAVteyZnxEQ8= -
 v0.6.0 h1:FXICGBZNMtdHlW65trpoHviHctQD3seWhRRcqp2hMOU= -
 v0.7.0-alpha0 h1:jNUuiQA/JPwq2IcIdwxbJ6kGsXvLs2ziOtJMBAu53Ms= -
 v0.7.0-alpha1 h1:a3TVZWNd0f5Ml+l8CQC/KzzbGBVP96wXHIft1QgvWuQ= -
github.com/coopernurse/gorp
 v1.2.1 h1:xpP6tTdvqeKvKEEKy3dj1r1+ubwYiwGRpYeXDe1Mgno= h1:wkfIkQktc4uuBo0kLNE8tMMN9okbsTa2orfZvBaL9F8=
 v1.6.1 h1:U51KxtIsUzG3mRR5QXNIfvSTyw9tUlZOjwsvgnLft+w= -
github.com/coreos/bbolt
 v1.3.1-coreos.0 h1:1OdDLDNDDBL667hjU02PnF6erC/bdeXAtz9IQwp29vE= h1:iRUV2dpdMOn7Bo10OQBFzIJO9kkE559Wcmn+qkEiiKk=
 v1.3.1-coreos.1 h1:crb2YOmpHMVsND8Ug24uvqgL8rUmJlQc+UVfUlMHalc= -
 v1.3.1-coreos.2 h1:E6ftWB6kbWjgCS2eQojRIAUsmCnoWPiYVKDu4GBsvtE= -
 v1.3.1-coreos.3 h1:sP70znHBV8469pbVsmR2G6wvd2oQwgxtgWyZvV3KVBo= -
 v1.3.1-coreos.4 h1:8bk3J1/F3lXGhVLhWkFutY4DTJeD7GOPBHt1rChv+QI= -
 v1.3.1-coreos.5 h1:e1dvLp11PaS48015lANrxa1aest1AkpEi4Hx/eLQQ3U= -
 v1.3.1-coreos.6 h1:uTXKg9gY70s9jMAKdfljFQcuh4e/BXOM+V+d00KFj3A= -
 v1.3.1-etcd.7 h1:QC4+fs/lXHXgUdHy3y4FPhky0OSvgsSPlNsxkLbX/C8= -
 v1.3.1-etcd.8 h1:xTcsP8rG1dLB1VRhYSyf6sFQnIU39vC7OkbtEU8bWIA= -
 v1.3.2 h1:wZwiHHUieZCquLkDL0B8UhzreNWsPHooDAG3q34zk0s= -
github.com/coreos/etcd
 v3.3.3+incompatible h1:moXyafqr1NI/E1v3IVz8BovXBjQKa1Oqwtsz7Lg38sM= h1:uF7uidLiAD3TWHmW31ZFd/JWoc32PjwdhPthX9715RE=
 v3.3.4+incompatible h1:jkDvc1ZyHw2mY2UsX66y+GHKPIIN7RhdBRDsr8m94Xg= -
 v3.3.5+incompatible h1:0qcOHgI5inKvd8lfaOD3KDC8QH2JQC86a1n81KXkRS4= -
 v3.3.6+incompatible h1:4tm4wgWV5rLFlAV4K6kj2zhSBH3bWvzQUyn88VgN3fE= -
 v3.3.7+incompatible h1:2uTvFq+hQI6TPmBUJMBbqeBy0RcEkKFw67tmK56xD7w= -
 v3.3.8+incompatible h1:uDjs0KvLk1mjTf7Ykd42tRsm9EkjCQX37DAmNwb4Kxs= -
 v3.3.9+incompatible h1:iKSVPXGNGqroBx4+RmUXv8emeU7y+ucRZSzTYgzLZwM= -
 v3.3.10+incompatible h1:jFneRYjIvLMLhDLCzuTuU4rSJUjRplcJQ7pD7MnhC04= -
 v3.3.11+incompatible h1:0gCnqKsq7XxMi69JsnbmMc1o+RJH3XH64sV9aiTTYko= -
 v3.3.12+incompatible h1:pAWNwdf7QiT1zfaWyqCtNZQWCLByQyA3JrSQyuYAqnQ= -
github.com/coreos/fleet
 v0.11.2 h1:OpO15BcPsSYYusarRQVd8Fl2eLKq1OgnxEfFF0dyvSo= h1:dcu9wb9bWRsaZDUDBAyl/qApKloFe+yoeC3M8dV6FYA=
 v0.11.3 h1:j5Tqk2KF1KQ4qeR87eZ1xIwr1DP69wyMeANiiISlDyY= -
 v0.11.4 h1:zDBez56EEqyAHx/f5AJMZexwSqOG0nObI9rYkFjMGkE= -
 v0.11.5 h1:Qu1zVN5dOKmzpDn40CzF4nXk+bRpQuc6CJ4MnuIsJkA= -
 v0.11.6 h1:NFx0oBUAZmhwfnsFG9el7Xc3DytgpQ78vhuwU5HmH+4= -
 v0.11.7 h1:BlugR/bvv8T8o0uW9wZZvQ4Z6ZkK7PnxaNYJzTFgHEI= -
 v0.11.8 h1:xzEWEiTUhsbu2Zk7oHmvLFIC9spf3o2MSqFSW18bQK4= -
 v0.12.0 h1:Df7DI3LOW26dh/6nyX8dnCA7BBFFkO0q2jFM5fa/ScI= -
 v0.13.0 h1:hQFVk/fbltdhFTPR0lMqwn++J5VKUCURv1+2Va+prGw= -
 v1.0.0 h1:xyIHQvFdFW3G36HL3qMWsQLQ0OWbh7KvcRrwqF0dNeg= -
github.com/coreos/go-etcd
 v0.2.0-rc1 h1:fGMM9WvZ0vZe5mRiPLMkAII4h8XSPYcldd2d/Gja6Oc= h1:Jez6KQU2B/sWsbdaef3ED8NzMklzPG4d5KIOhIy30Tk=
 v0.4.6 h1:v7gkaX+WwHRnP4Bl4Sc6NaYNa5U7VMDpFqwPMbXWXOE= -
 v2.0.0+incompatible h1:bXhRBIXoTm9BYHS3gE0TtQuyNZyeEMux2sDi4oo5YOo= -
github.com/coreos/go-iptables
 v0.1.0 h1:Vb3SuBct2T4LtfA1VASiDhE4rVMvwRnEhlKSqFb0YvQ= h1:/mVI274lEDI2ns62jHCDnCyBF9Iwsmekav8Dbxlm1MU=
 v0.2.0 h1:RmVRALeVCicZcF3rF05e0ooU9x9TmalN0HcT4hkhG5s= -
 v0.3.0 h1:UTQkjHl9rPwwtXZhXbY3T932cV9aUnKlSsZ7YGfJVXM= -
 v0.4.0 h1:wh4UbVs8DhLUbpyq97GLJDKrQMjEDD63T1xE4CrsKzQ= -
github.com/coreos/go-oidc
 v2.0.0+incompatible h1:+RStIopZ8wooMx+Vs5Bt8zMXxV1ABl5LbakNExNmZIg= h1:CgnwVTmzoESiwO9qyAFEMiHoZ1nMCKZlZ9V6mm3/LKc=
github.com/coreos/go-semver
 v0.1.0 h1:/7eIWhjNpsljQCSU+5NaZzu7HmMvoWnMqOmNZljz1f0= h1:nnelYz7RCh+5ahJtPPxZlU+153eP4D4r3EedlOD2RNk=
 v0.2.0 h1:3Jm3tLmsgAYcjC+4Up7hJrFBPr+n7rAqYeSw/SZazuY= -
github.com/coreos/rkt
 v1.22.0 h1:MiJBjcLSRJgB86SfyqD+LpCynNuBenCquj11xi5IKd4= h1:O634mlH6U7qk87poQifK6M2rsFNt+FyUTWNMnP1hF1U=
 v1.23.0 h1:ehH3hL6aqgFiAuDGVBrFVNtBf/6khdjhDPoCRbAmUlY= -
 v1.24.0 h1:v6JVSfu8X3r9pJX6tYwdIs2LS0o76E9aisXkrQcuNEY= -
 v1.25.0 h1:4rDjbXDtLwReumvn9/fH0D4aTXWXk2kpUcpjJNH2F/g= -
 v1.26.0 h1:W8o+Yq/nKrqDOgtAfhusAosi11C5EHSFwIho02Qq4E4= -
 v1.27.0 h1:OPLnOwaV46mif6IjKGjL80HYq96pfqyLfUeJKTYjGRE= -
 v1.28.0 h1:EUsbk1zTx5jF3uEQzkWK/gAfOv0xyRDsKhIbtmJPHgI= -
 v1.28.1 h1:v2Nd8Zn5GQen/F+GCTn7cWRsdB/HBMOSpEO35f4zwog= -
 v1.29.0 h1:YEVlrEX/ZW8n8tHHSuPbDOPWewpSCIF1DcHw0nHPEXM= -
 v1.30.0 h1:Kkt6sYeEGKxA3Y7SCrY+nHoXkWed6Jr2BBY42GqMymM= -
github.com/cpuguy83/go-md2man
 v1.0.1 h1:5hPj0PAocb2UrJ9Zhr36Qw+Bd+D/YdcwKFj2pT9ENkU= h1:N6JayAiVKtlHSnuTCeuLSQVs75hb8q+dYQLjr7cDsKY=
 v1.0.2 h1:VVmqbsnngY4dm7hPLp+fLYx5M3B2NgBp+7oQNVuNAv4= -
 v1.0.3 h1:IPx6io3Yvb5JMk49BHkvmTMNYid+Z9YsYL/V8ROC418= -
 v1.0.4 h1:OwjhDpK9YGCcI5CDf8HcdfsXqr6znFyAJfuZ27ixJsc= -
 v1.0.5 h1:1T5gaHeuXe4gAhG/ZKRPd4WbbdEFOVYkNGTZ13+fktU= -
 v1.0.6 h1:oNbTgn74tKz6PzZM2Zpmm04rMdAleoj9GrYC3SCKiU0= -
 v1.0.7 h1:DVS0EPFHUiaJSaX2EKlaf65HUmk9PXhOl/Xa3Go242Q= -
 v1.0.8 h1:DwoNytLphI8hzS2Af4D0dfaEaiSq2bN05mEm4R6vf8M= -
github.com/cznic/ql
 v1.0.0 h1:W2I+yOLLPgrUh0knGBsvnc/d2V4TeWp9xBbI8xFLVdM= h1:YhViR8X3dW5uRADlFXxbh3suMUVYoCeAdyhweCWukWk=
 v1.0.1 h1:9R6NwTYlIfpon+LNndB3eRZ5HuMczHlpgF/l5VxqfLA= -
 v1.0.2 h1:2N2Vdj48/tMjBymxTXVKa2XsAWsZREifAnHKxdUiY7U= -
 v1.0.3 h1:fGNAaW9MolJmMjDm/3EUqdTmM/PPvrSNLt8qJh0zpNA= -
 v1.0.4 h1:LgBptPUTTk0mtWi+NZjNuQSbizMc2e88hUGoAjpy31U= -
 v1.0.5 h1:nh3G9w4iAqx71/h2XfQWM/IMj+I9Lw479phJ/e/Fcpw= -
 v1.0.6 h1:97U34dzh5uJAUmbkgt06M0tQH52//PRvl0q3n2HDpwY= -
 v1.1.0 h1:/ByVHV7ADDnAVCNzYvJGBnCYF+YR16qGPX4Q+tBRK4c= -
 v1.2.0 h1:lcKp95ZtdF0XkWhGnVIXGF8dVD2X+ClS08tglKtf+ak= h1:FbpzhyZrqr0PVlK6ury+PoW3T0ODUV22OeWIxcaOrSE=
github.com/dancannon/gorethink
 v2.2.0+incompatible h1:f96+JZnNpTm/cD3wz7f+IIdlXJlRRTU2Tq6Pu9RIDds= h1:BLvkat9KmZc1efyYwhz3WnybhRZtgF1K929FD8z1avU=
 v2.2.1+incompatible h1:Fj5UJxnECayN88GJ4nLcrsW2fORBJU7h8rG2l/o+CnA= -
 v2.2.2+incompatible h1:pNxbnZHZfuxsCXajrhfBTLVz7Psu3j1BDZGGa2moLho= -
 v3.0.0+incompatible h1:OOGAzliummoaEOFAiZTYH6EvNO5O/m6X8AjpfyfNht8= -
 v3.0.1+incompatible h1:O2gbvj46lA3g43hN2raM2aUGBgBOWGZnkN4vV45oaaA= -
 v3.0.2+incompatible h1:YkK7LB7m/CLigNDSLFZaWSmnRzf4GDpXDrEpuBFqpy8= -
 v3.0.3+incompatible h1:VFL9GPZ5DMdj34nviX7RG5PSxOtbA3XiAz+XuYcAnm0= -
 v3.0.4+incompatible h1:CbiS/bFQunLtttIgPO7U9miUBwlBp8PbqWIqbuFPD4U= -
 v3.0.5+incompatible h1:AenyUfDNo7CUhNsA9qGVDIO/b5GzrnwjBRLZvlkb7jk= -
 v4.0.0+incompatible h1:KFV7Gha3AuqT+gr0B/eKvGhbjmUv0qGF43aKCIKVE9A= -
github.com/davecgh/go-spew
 v1.0.0 h1:TJ+L3B1N2zmDTa+nwZ1QMI4Dn2H85x5QbmgUlksLY7Y= h1:J7Y8YcW2NihsgmVo/mv3lAwl/skON4iLHjSsI+c5H38=
 v1.1.0 h1:ZDRjVQ15GmhC3fiQ8ni8+OwkZQO4DARzQgrnXU1Liz8= -
 v1.1.1 h1:vj9j/u1bqnvCEfJOwUhtlOARqs3+rkHYY13jYWTU97c= -
github.com/dchest/siphash
 v1.0.0 h1:8nHH2JstXfT60brTp2MOmeYHgLesIB7f85o6XBKllGU= h1:q+IRvb2gOSrUnYoPqHiyHXS0FOBBOdl6tONBlVnOnt4=
 v1.1.0 h1:1Rs9eTUlZLPBEvV+2sTaM8O0NWn0ppbgqS7p11aWawI= -
 v1.2.0 h1:YWOShuhvg0GqbQpMa60QlCGtEyf7O7HC1Jf0VjdQ60M= -
 v1.2.1 h1:4cLinnzVJDKxTCl9B01807Yiy+W7ZzVHj/KIroQRvT4= -
github.com/deckarep/golang-set
 v1.7.1 h1:SCQV0S6gTtp6itiFrTqI+pfmJ4LN85S1YzhDf9rTHJQ= h1:93vsz/8Wt4joVM7c2AVqh+YRMiUSc14yDtF28KmMOgQ=
github.com/deis/deis
 v1.11.2 h1:YfAeyyIj8WC/W1LqYgQSiqe6Ne9WxbXqOhtcYB33EHk= h1:8wwLzvV6ikTNSIAIj8CTdkloCRGy7C3wMrkB3U00Nx8=
 v1.12.0 h1:4F1oOQak+ysbLH6eCqqYL7eey1c041QZJaGvB0/v4oE= -
 v1.12.1 h1:fxE7grqHynY2UDF9v/AtdhMcYRWGKwnM070mh+n7/UM= -
 v1.12.2 h1:e2r+XiCn/X9t7EtmJIny5itFPdrB7oM1e35i3gOf5g4= -
 v1.12.3 h1:EbDDhEYYRMEvyeSOgO+kzPjistPvrfpDiGHTW++wLMc= -
 v1.13.0 h1:/+S+PoBhd1KYot8BhrWaP60z17jDqnNHGxyxK1LAvG0= -
 v1.13.1 h1:WphZ3nF5n6oa27mRZoyRpdHLN7ztGqzNcSwRlNKOON8= -
 v1.13.2 h1:s/13DAN0CO6vJG7FiuzwdRjtnwC3lozaoLlZSBameek= -
 v1.13.3 h1:an/Fac3S2yc6ai8I7v6/f7qoFkjoBIaZMdOwwOrCXwo= -
 v1.13.4 h1:RApXfuncu0isEEIshdGQd2jmaBRRMSE7nLQVKK62zgg= -
github.com/dghubble/oauth1
 v0.1.0 h1:SsG7ISiFFawNGiUDr3+1FGoEvCxu/EBo0fkPM4H32OE= h1:8V8BMV9DJRREZx/lUaHtrs7GUMXpzbMqJxINCasxYug=
 v0.2.0 h1:QolJrVUMbkm4KqE/izyfcGDUjs0EqXz+BLeuz/wVQ6c= -
 v0.3.0 h1:tEj05GFDD47m1ZD7XPwBTcKi4KgKxhwNImmoREbxAbg= -
 v0.4.0 h1:+MpOsgByu02lzT4pRGei5d2p6nAgj8yDwF3RASrgNPQ= -
 v0.5.0 h1:uJqX7Rzr3QRmp2slUWqI9Sm8NoP65AMiyXiijOWWLvQ= -
github.com/dghubble/sling
 v1.1.0 h1:DLu20Bq2qsB9cI5Hldaxj+TMPEaPpPE8IR2kvD22Atg= h1:ZcPRuLm0qrcULW2gOrjXrAWgf76sahqSyxXyVOvkunE=
 v1.2.0 h1:PYGS9ofwbV9nfhB1kYjB1vtXshMxlp2oQxTMMXVJ5pE= -
github.com/dgraph-io/badger
 v1.1.0 h1:NB8GzptcU1D5Lf3KNYIchxtDPaduDg/5LYO7TnnC74A= h1:VZxzAIRPHRVNRKRo6AXrX9BJegn6il06VMTZVJYCIjQ=
 v1.1.1 h1:P5UKapWWkEE/3A7fnJ5AX2AAEViIjlHjbgR/PgqXmNs= -
 v1.2.0 h1:FpzZXaTZEP0EtRtxrSNVHpK7G7Rep6N5JAvBkLOWrcM= -
 v1.3.0 h1:C85TWgcBeo91E85jbWwz2tM/UgcH0F/aQJL8svIccCU= -
 v1.4.0 h1:3W68dJQBX97hQNZk5IO4gN8fDMOSCy6MD2F0EXzeIPw= -
 v1.5.0 h1:Iq6peB6gz+ABCQQuxa7lOj9j+knLvMYjyhydw8fiSxo= -
 v1.5.1 h1:B2S9OJ4JNKBLcc/0rzklp7wS6yjytvsTekWE/7YZRa0= -
 v1.5.2 h1:nIdKlAGfMD8n1hU0/7CK9KhtuJt16BVmlGSDBNOLCQw= -
 v1.5.3 h1:5oWIuRvwn93cie+OSt1zSnkaIQ1JFQM8bGlIv6O6Sts= -
 v1.5.4 h1:gVTrpUTbbr/T24uvoCaqY2KSHfNLVGm0w+hbee2HMeg= -
github.com/dgrijalva/jwt-go
 v2.1.0+incompatible h1:S8T+eIrll7F0MrbgOO7jWvVmP1B2CsLwFvjXPOVM8Jc= h1:E3ru+11k8xSBh+hMPgOLZmtrrCbhqsmaPHjLKYnJCaQ=
 v2.2.0+incompatible h1:uxMuTjjks/pNHqM+CN0IGKK9ZSs3j4NWZFnEm/5yGoY= -
 v2.3.0+incompatible h1:GS7vCh4X6j+n3crClgYSu5RjPBjd9iHeQ/Se3i2q7h0= -
 v2.4.0+incompatible h1:5PCcRjlXqAvUY9W5yoWUyUp3i+h07i8RCn2s0V959Ic= -
 v2.5.0+incompatible h1:k57pRbmSsdonRfxM9oTC2fR40mlKHsuKNZ3UlrY7u24= -
 v2.6.0+incompatible h1:1O2NahKrpPnTN8Jy7ffR7S31wnTZIM4PEzrF8br1R4g= -
 v2.7.0+incompatible h1:54T2qn/iIwjg7JGrMsKD3WID0+CaYUrJgyXDM5ckYLk= -
 v3.0.0+incompatible h1:nfVqwkkhaRUethVJaQf5TUFdFr3YUF4lJBTf/F2XwVI= -
 v3.1.0+incompatible h1:FFziAwDQQ2dz1XClWMkwvukur3evtZx7x/wMHKM1i20= -
 v3.2.0+incompatible h1:7qlOGliEKZXTDg6OTjfoBKDXWrumCAMpl/TFQ4/5kLM= -
github.com/digitalocean/godo
 v1.3.0 h1:skAw84H7ScMEoFJk9ASHlJz6A0Bk48Us7K04e7uXwOo= h1:h6faOIcZ8lWIwNQ+DN7b3CgX4Kwby5T+nbpNqkUIozU=
 v1.4.0 h1:W0WhkDf627pKcsjdAJ7kmViGoQePnLdqyXJwO2OxN3A= -
 v1.4.1 h1:mm4PBdiicsWxrywnCNmoBGV90CEqHcb+j994FGQ4NVE= -
 v1.4.2 h1:AOsEb4Sa2OpfIAZluAvA5LGp61uq53tCNs+BZte1I28= -
 v1.5.0 h1:S2HcGAP6RQ46tXUmQ6lTJP+HzOGB9XvCxA85eccb3Fw= -
 v1.6.0 h1:BjFbURUz1LDimGfApIWqzZiSrJFbtyGYOHfSRq/Voi8= -
 v1.7.0 h1:JIh+VoKGbK83feLAsgx5IPMG2VHK5/YjFVK14yDXEPA= -
 v1.7.1 h1:bc4MQyL5ou0bsIx3PcVRDO15xQyyXFbJjE9aE6H0QPc= -
 v1.7.2 h1:n11Hu42Gz2jG+IOAELoQS8zlZfPG0N/lo/k0DtdYp+4= -
 v1.7.3 h1:0tFPilFBDsVSiAKF8hyzj/MAmRYxNc5MGhAiYZYVrpo= -
github.com/dimfeld/httptreemux
 v3.5.1+incompatible h1:Sfytas3Oljs5RAkfdWG6jDFSBjtfI1fDhFxib/2iqOg= h1:rbUlSV+CCpv/SuqUTP/8Bk2O3LyUV436/yaRGkhP6Z0=
 v3.6.0+incompatible h1:jteMiDNtk9+Cp+WmdLaQ9OUw0lF/EFhwi/imunkcDmY= -
 v3.7.0+incompatible h1:cuOxlKQM3w56FmqBmgECAiDT/yTiCBftcGbCGEZODWg= -
 v3.8.0+incompatible h1:GkWWdeNjnUXeTXPMqNyC8VLXrZbHLVRB1+kpI2viEtw= -
 v3.9.0+incompatible h1:RLjwORnEXePoxnZsEbSdOW1MnogylIzckiBIOAjx3gY= -
 v4.0.0+incompatible h1:e+RyY+i9Z77Ww015gYKYJhWIQXvPHJOFgzj5XnWcVY8= -
 v4.0.1+incompatible h1:q52LJ/5vhen7pnv/LmpyTZuALCQuyyNrl1MWpuI2yiU= -
 v4.1.0+incompatible h1:atId6gcRAedcZnVuaNuyTOMu/L+U2MCRaqH3kn6yNkg= -
 v5.0.0+incompatible h1:WnFKlZjOBy5nNALaVfT/yG4XCzxpHPQ96C/u8AP+x6s= -
 v5.0.1+incompatible h1:Qj3gVcDNoOthBAqftuD596rm4wg/adLLz5xh5CmpiCA= -
github.com/disintegration/gift
 v1.0.0 h1:whtptdjB6gheQRwEQ8K9a88IGELuyxYuJFojBfQLpRI= h1:Jh2i7f7Q2BM7Ezno3PhfezbR1xpUg9dUg3/RlKGr4HI=
 v1.0.1 h1:CnLD4dFNei6R+1bGbtR9r4z1HO7rqIi7+EgZWQY/gC0= -
 v1.1.0 h1:q2eJmb3SWqnFH9GonP+K9P06Jk+861UKol5/5juXxz0= -
 v1.1.1 h1:T0hwEROfntxovpHD7rokgRNx18see5mwdx4PagQ4oZQ= -
 v1.1.2 h1:9ZyHJr+kPamiH10FX3Pynt1AxFUob812bU9Wt4GMzhs= -
 v1.2.0 h1:VMQeei2F+ZtsHjMgP6Sdt1kFjRhs2lGz8ljEOPeIR50= -
github.com/disintegration/imaging
 v1.2.1 h1:2sSJ9O5zdLUbo1H5ul1hxEi/s7uMfCBVXRzzVXp3Gwo= h1:9B/deIUIrliYkyMTuXJd6OUFLcrZ2tf+3Qlwnaf/CjU=
 v1.2.2 h1:Vm71h4WNvlNbmZSerkAD0Qqg2sV5Ojedd8+0u+cjbO4= -
 v1.2.3 h1:OjtUcbVpC/wqPdyUzOdFFF0Ewj1+5hsdYxwGiE68BKg= -
 v1.2.4 h1:eJRPGef+mQ4WZ8cED/pqElxW4+79zBjJYTjYv48GZOM= -
 v1.3.0 h1:AihaBC+K3RHP7JL8scqq2GgNQhVzMcCYh7gJbAjAvpo= -
 v1.4.0 h1:iQR5SpN9kQx8/8Cs33yDQUCRY4aEtBX8en6o6IYS/oY= -
 v1.4.1 h1:cwtUiS5AYdPT0/GRE3EFHnT+ajKBaYsqzPKOMmqqudk= -
 v1.4.2 h1:BSVxoYQ2NfLdvIGCDD8GHgBV5K0FCEsc0d/6FxQII3I= -
 v1.5.0 h1:uYqUhwNmLU4K1FN44vhqS4TZJRAA4RhBINgbQlKyGi0= -
 v1.6.0 h1:nVPXRUUQ36Z7MNf0O77UzgnOb1mkMMor7lmJMJXc/mA= h1:xuIt+sRxDFrHS0drzXUlCJthkJ8k7lkkUojDSR247MQ=
github.com/docker/containerd
 v0.1.0 h1:TMb+CQk29b9dAOwux0UQhgP85/tC5Mm6OrI5itCD7+4= h1:yNozov3Y4G0QO25BR7CwoSEGPvBQLXitaauXtxebl+Y=
 v0.2.0 h1:uWiAs1d94h7pJ8hGxilTP4NSWvcpsSQhII6zcNMNkog= -
 v0.2.1 h1:Cnev0G9QmUUXrBnMtYaWEmI8ulRro2Q2GCFyxGq3WPE= -
 v0.2.2 h1:oq/n4rFTtQ0YTY3incyeMhvPo2e2XxoLKDfIbLAA3yw= -
 v0.2.3 h1:h9L1a6cCk+3wOeNeBt+DJxziJnRBpj/27Ja9yUY6g2E= -
 v0.2.4 h1:kcPq4DBVt6mSf7YOKwo+g/oyLVZ5+CjAPzfvDWaWz88= -
 v0.2.5 h1:QhtWftJ75KdmBMJ4/0ctPGfPr1Gfs5vOk7QXRX0v3p8= -
 v0.2.6 h1:bCcR9drrwxWpOXWO5peqo6gm5LiKNy78WOW/g2gwuUs= -
 v0.2.7 h1:FjHCZfjUmFkO5xLl/xDqu4Uaq0CYXZ1KkZKuUHyMRrk= -
 v0.2.8 h1:Uh3MiqAr3PUVT12rgBQpwZLfV/lvu8D7hVkrJ/evuWA= -
github.com/docker/distribution
 v2.6.0-rc.1+incompatible h1:DdOu8ihP6OokkRb9mcR6zfrs7qZFQYMQubUkRNwxaCY= h1:J2gT2udsDAN96Uj4KfcMRqY0/ypR+oyYUYmja8H+y+w=
 v2.6.0-rc.2+incompatible h1:EamBfeQ+My8FWfOANNc/JdV7LAyU256HbBoO2hLo04w= -
 v2.6.0+incompatible h1:8hxvSzNSatS/Ik5y595JXjKnUjd/Si3lMM8irrl34PU= -
 v2.6.1-rc.1+incompatible h1:8b9G8zGEzX0X7ddZWJBta+LPx0P3FOmUyzeD2LU0MQc= -
 v2.6.1-rc.2+incompatible h1:/XKDQO2mXt96NI7anU/cGhLbir5zJif/vD1WQXGZUTI= -
 v2.6.1+incompatible h1:nWLG5KfUJK1PBt7i7iLBE8KpgWVF4uXwNfK1YOi1K/I= -
 v2.6.2+incompatible h1:4FI6af79dfCS/CYb+RRtkSHw3q1L/bnDjG1PcPZtQhM= -
 v2.7.0-rc.0+incompatible h1:Nw9tozLpkMnG3IA1zLzsCuwKizII6havt4iIXWWzU2s= -
 v2.7.0+incompatible h1:neUDAlf3wX6Ml4HdqTrbcOHXtfRN0TFIwt6YFL7N9RU= -
 v2.7.1+incompatible h1:a5mlkVzth6W5A4fOsS3D2EO5BUmsJpcB+cRlLU7cSug= -
github.com/docker/docker
 v1.13.0-rc2 h1:816gJHhXHI2MMuvNI0oUtA4tOxqcTWYjtYwqSr7a9Fo= h1:eEKB0N0r5NX/I1kEveEz05bcu8tLC/8azJZsviup8Sk=
 v1.13.0-rc3 h1:p9v49W4qKQmWR6vZl8jfvAYRyjzVLv0z3dHgA7RNO88= -
 v1.13.0-rc4 h1:mxHN7wjGO/ogV1HaNZsaYWzed+NLlPSPfKhup/xqpho= -
 v1.13.0-rc5 h1:J5Ztzo7583rNEXBaavIyz1tcNB7oP9md7OuNUvUEV3U= -
 v1.13.0-rc6 h1:G/yPCx94tZqlP+5eguVY2ieDFWQvfMoHCRjyl+RIkCs= -
 v1.13.0-rc7 h1:DRywiGPrXjuceK3YyB0sxreg9TduviYZvSSh5G1jS2Q= -
 v1.13.0 h1:A5bSnWdwEvPOZ72/0cMjcYIyuvLCHk/KVrT8qOwdKkw= -
 v1.13.1-rc1 h1:TRvGNDqqo/e7CArgPblEI+pvUFHM4in1UPtuwuwO2N8= -
 v1.13.1-rc2 h1:4cP9ZpIRaTsEECeg+lOpR0Txm93gixo49VjuI+E+v/s= -
 v1.13.1 h1:IkZjBSIc8hBjLpqeAbeE5mca5mNgeatLHBy3GO78BWo= -
github.com/docker/docker-credential-helpers
 v0.2.0 h1:KtTV34fmVBrwF2H96c9/MmxRhMneVDxlNbwlrH0aN1k= h1:WRaJzqw3CTB9bk10avuGsjVBZsD05qeibJ1/TYlvc0Y=
 v0.3.0 h1:FOPAhzCCjbiZTs7B1m5REioFVT/srLnFXiSeLO128c8= -
 v0.4.0 h1:6vP4fWW/rUUyyBlT+DeZi5ueMWp1n4YRkT1kzYC7Ls8= -
 v0.4.1 h1:ctHYuzQopMGaPgNpgBn0xiY/TqVcWjjdprR8SSsF1PE= -
 v0.4.2 h1:SADvzUM5kYy7P38edX2cFkri5UnKyS6HCrzfC0V0xUA= -
 v0.5.0 h1:2B+Y8H9kp4G1UGRWWxxNz0D0aXVHUA3RCnI2/Pm4xNI= -
 v0.5.1 h1:57nLGp6DCMQTz4Otl2bWX7MY0wcuQDk9EQrX0+OpF6M= -
 v0.5.2 h1:Zz3b++zosJsPIy1Rnus+4LO2T1Ot5BjC3eT1Qk9XFjE= -
 v0.6.0 h1:5bhDRLn1roGiNjz8IezRngHxMfoeaXGyr0BeMHq4rD8= -
 v0.6.1 h1:Dq4iIfcM7cNtddhLVWe9h4QDjsi4OER3Z8voPu/I52g= -
github.com/docker/engine-api
 v0.1.3 h1:jklhkNWOIpdm33EDXZbS5lK2iffmxkyB0GL7ipQPxGw= h1:xtQCpzf4YysNZCVFfIGIm7qfLvYbxtLkEVVfKhTVOvw=
 v0.2.0 h1:QAxolvM0QMlJK1RawCNNAAdFTpZBb7hE2Ep2fpdm2S8= -
 v0.2.1 h1:fnDU2LrRVcZvaaFws95ekr/i3NRH3bFPEgDSXhV7oLM= -
 v0.2.2 h1:49f9D03xUALK02A7iS0mnYd5F0oJ5AynQFzyESdY9tQ= -
 v0.2.3 h1:eRUM6H5g52nZqFrrXFRPRKNbz/UmcLM2EgqUD8Zwe/Q= -
 v0.3.0 h1:jeex4VYn27z2DFCnL8vcWld/o46IgcT3bIj3DlOAqJk= -
 v0.3.1 h1:d50wl50my0c35Acjdm2AidAW7XU7dbWC79Ue9r0aNU8= -
 v0.3.2 h1:gRSoZwtRdgDtOOK1mRfYNGoq+LaUTSAMN6FylqhMZyA= -
 v0.3.3 h1:+qfDCFf8WGX2s5UgerD5BZtxMC8LySq4c6nH/mrwyn0= -
 v0.4.0 h1:D0Osr6+45yAlQqLyoczv5qJtAu+P0HB0rLCddck03wY= -
github.com/docker/go-connections
 v0.1.0 h1:CYk/56klHjBmLlbozTT8SPTZc/Wetw1IKu168WUZi5w= h1:Gbd7IOopHjR8Iph03tsViu4nIes5XhDvyHbTtUxmeec=
 v0.1.1 h1:90BpDo1ETO0J9w7hKZkCDRrjbHniqjsy+VaP7MbLEN0= -
 v0.1.2 h1:X2kEYuCRBYv9hH68gobBRGZhMGcEBMDgGHtavCfwUyE= -
 v0.1.3 h1:1B+oHEtDXzSjbetalX+YNkw6qV1/uOybnv27B7FQDTY= -
 v0.2.0 h1:tV+S3i76CmPRYmR3NMDUFyr2HTP+3gL+xEPy146TPig= -
 v0.2.1 h1:XB0Pr+bR+RGw8D0C/ADeRiiPVyMftTtKFblUw3sNFXQ= -
 v0.3.0 h1:3lOnM9cSzgGwx8VfK/NGOW5fLQ0GjIlCkaktF+n1M6o= -
 v0.4.0 h1:El9xVISelRB7BuFusrZozjnkIM5YnzCViNKohAFqRJQ= -
github.com/docker/go-units
 v0.1.0 h1:LBCjdXuTJjjlJU5Lg59URp4ubf693pHgo2Dvm/P035o= h1:fgPhTUdO+D/Jk86RDLlptpiXQzgHJF7gydDDbaIK4Dk=
 v0.2.0 h1:TtZVwKVMsN8COBXUhH/x17NFxEFfIIK2i9DL/nz4zfE= -
 v0.3.0 h1:69LhctGQbg0wZ2bTvwFsuPXPnhe6T2+0UMsxh+rBYZg= -
 v0.3.1 h1:QAFdsA6jLCnglbqE6mUsHuPcJlntY94DkxHf4deHKIU= -
 v0.3.2 h1:Kjm80apys7gTtfVmCvVY8gwu10uofaFSrmAKOVrtueE= -
 v0.3.3 h1:Xk8S3Xj5sLGlG5g67hJmYMmUgXv5N4PhkjJHHqrwnTk= -
github.com/docker/libcompose
 v0.1.0 h1:iDj0n0W2Zw8+hIE4ftGYMQvnulxzmmvBIUzrieffLT0= h1:EyqDS+Iyca0hS44T7qIMTeO1EOYWWWNOGpufHu9R8cs=
 v0.2.0 h1:qC7yKsuR+ihYQdqSbBM80rT426kHZ1XIWFOPOyaStxM= -
 v0.3.0 h1:iFImvNns7hpqM2Wtdv3gn68t/ojorjreIAJNHs/pOms= -
 v0.4.0 h1:zK7Ug0lCxPB8FDFNdCvR2ZjJjeJZ/607lfAYkp1hrtc= -
github.com/docker/libcontainer
 v1.0.1 h1:qym49STGNmOFO+8lA0PXgJFbscoZElKFeGE0BPAjFKA= h1:osvj61pYsqhNCMLGX31xr7klUBhHb/ZBuXS0o1Fvwbw=
 v1.1.0 h1:HLZTmM3/fVeR4V5S8NAF2t44zCCIzJ/ZL/jRe9wPBOU= -
 v1.2.0 h1:Gvq1FDdzhLhC2TIJ7jtP4k2ba9J7dXd+Yron9QMNy8c= -
 v1.4.0 h1:mrCCYnQU1lnp43y/1ViKbGjucA+Txbmf7hPIdC8+r7A= -
 v2.0.0+incompatible h1:Tyr3weHr7zERwvkOgDnB4346Wrn72bePTI5TspHLpf4= -
 v2.0.1+incompatible h1:r4m0Ya1uFdkFg4dcRbFCzBBOr8bC7j/J0ZLCtJR3UNA= -
 v2.1.0+incompatible h1:0gatB5yQLIq7gywAS0HmXOY22/DyMIkW7HVhc/jOOsY= -
 v2.1.1+incompatible h1:gXUagKm91lZouHqeObbFBZJJNjZWFB3NJ+eaTAjlOaI= -
 v2.2.0+incompatible h1:0FYWwtMBwDN0lMTugJBZWc4sG25prVGIO1Sc7XmHXGY= -
 v2.2.1+incompatible h1:++SbbkCw+X8vAd4j2gOCzZ2Nn7s2xFALTf7LZKmM1/0= -
github.com/docker/libkv
 v0.1.0 h1:3drKYG6raiVY90exyMbehg9WGVXfLPyUeVmNphkMw3o= h1:r5hEwHwW8dr0TFBYGCarMNbrQOiwL1xoqDYZ/JqoTK0=
 v0.2.0 h1:Jvz/q3IEbCsr7rWVXUTN/riH7wMBWJxZLvW3yyc5eJA= -
 v0.2.1 h1:PNXYaftMVCFS5CmnDtDWTg3wbBO61Q/cEo3KX1oKxto= -
github.com/docker/libnetwork
 v0.7.0-rc.1 h1:Uisi0o6s3hEY8l8khJMamp++4LK/iMKhwpxvtP8dspw= h1:93m0aTqz6z+g32wla4l4WxTrdtvBRmVzYRkYvasA5Z8=
 v0.7.0-rc.2 h1:Ig+jZrXXIZDzdPOTQ4wv/48kNA+vq+/BhSi+93hAkto= -
 v0.7.0-rc.3 h1:b/ngW6LcjkX1JZUbXx40Za1seXfEKlZ7xQsTqOK6gbs= -
 v0.7.0-rc.4 h1:jleug9AP1rbzblwowJ7bH+JMyJoS7bPYUSUTTQ1308Y= -
 v0.7.0-rc.5 h1:lXCsgiubX4bYnXwtEcaJqYtuy8nEvRTVStvOTbxFWKQ= -
 v0.7.0-rc.6 h1:IzRM9DwL+4TZK5wrAXEfy+4ZbarMZ2EiT3skD3DJNuw= -
 v0.7.0-rc.7 h1:goL5cAJK0qwO2mI8J17S+V+YUcdw6aHyz9lXqhitBXI= -
 v0.7.2-rc.1 h1:9QhbopyfyT9fjlYtB8+knmX9o3mQdoQ7jQ/guDUaaUM= -
 v0.8.0-dev.1 h1:fTxjpWMDedJgM3ohvKqVAk9HW6FulRPIvSiPS3Hgljw= -
 v0.8.0-dev.2 h1:1N717njYC/nF66PoDUPAykdeCGFXCUkB6SHXtEYTKpE= -
github.com/docker/machine
 v0.12.0 h1:vJc/Lcw/cNeS8kRyPXQ73qlZftP77nn4wahg58FXdwQ= h1:I8mPNDeK1uH+JTcUU7X0ZW8KiYz0jyAgNaeSJ1rCfDI=
 v0.12.1 h1:uo7xAesXJTkkreEIOCBhVirPsvkhaVxxOFHWuIoaxqw= -
 v0.12.2 h1:hnEiFuJKxmals/98skBZ5Oq7aWU1Ah3YKMUluQHvtLA= -
 v0.13.0-rc1 h1:0ylLk7OT4oelNuBWQzp6DCo+w8rOq2jL8sNGtHolMAU= -
 v0.13.0 h1:qXOQLfafxknrwNNa9PNOLcFFqjb/rid9susqEsl6Xs4= -
 v0.14.0-rc1 h1:kwk4/a9cP/kcuXw10nd4/inCLOLcFSBl7q/KnVSlcvY= -
 v0.14.0 h1:7zMwMAz0XbL27Q4NPqtW7I0wuhOaGFpq5WfvrUfZROM= -
 v0.15.0 h1:4KK/T1Pyb40vYI1ugMayMAEhkaNfqZekLaHPLF4lFvg= -
 v0.16.0 h1:nLI3xyw6MD6qgw1e3Cv/oyW6qTpioU+SSDxoQQ5Mw+Q= -
 v0.16.1 h1:zrgroZounGVkxLmBqMyc1uT2GgapXVjIWHCfBf0udrA= -
github.com/docker/notary
 v0.3.0 h1:ly6/JWRw7Edho8YTNi6KSr9RJRXVRaSn8vtbJ8aD9X8= h1:3/NLyxebcX4foXQ9v90i88wEj1B3rsq6aVhtQIvYFe4=
 v0.4.0 h1:sIPZtvPFtkEo+y3krh4CO7IQPCbNhV2BXw6BxrkoCWI= -
 v0.4.1 h1:uyRDPTOcsy8Esd194cuxq4KQz2v/YaevNi41Pw6Zv7o= -
 v0.4.2 h1:9UGXtaXyuU7r+TjRtgBdRJWtEhdrsn39vmRXulK0JXc= -
 v0.4.3 h1:pbU2M0r4F5lcku1CCsw+pvBU/1EMUktM0kBqVl1+irs= -
 v0.4.4 h1:Acz6MeXhCHJk2D47zXn+ol0rn1keC8KMxcxPspH0JnE= -
 v0.5.0 h1:pu/2hwX8VBwy0ABV8n35AESUN1GCIcf9laD/8elq8wg= -
 v0.5.1 h1:L+2sox5zXEfpITxZLZJyW8gw2M+XNnggoSr959pR4lE= -
 v0.6.0 h1:bpHW1Bh7O8XZXvDSnybuiu971oWTNXsaygbJt1qlcS0= -
 v0.6.1 h1:6BO5SNujR+CIuj2jwT2/yD6LdD+N9f5VbzR+nfzB5ZA= -
github.com/docker/swarm
 v1.2.4-rc2 h1:Vm3E9Az2DRBlhp7fdCOMPYwZ3OZwcioPJt98GvcauPo= h1:rcMN9vZajPn8s2MgNl/C9G2nUaJrCMHIWvliOrQ4SJ8=
 v1.2.4 h1:ng7K7fBoFBOfemhQkBHQGIgya9ADXoiQdv5aJtEr5c8= -
 v1.2.5 h1:7ZUm1hpv9DuD/2a8v0mC+ZX3QN0U7ToPQzhxKQk4HVo= -
 v1.2.6-rc1 h1:Fo2L3vz/JmIE2OtSEhJkklxaSPira6cyvuycktsSCV4= -
 v1.2.6-rc2 h1:wzLmvtXo6O75LIsEAozXPKw4kkRZW9GkDciFnAnXqPI= -
 v1.2.6 h1:5AGZMVjRoPyS/b1FTZe7Wfk3dvdnDBDlvQg323K8mpg= -
 v1.2.7 h1:Qd1HrDvjVJWcFZOv+XjUaNNMa24CzwaSTSH2RiWQcd8= -
 v1.2.8-rc1 h1:TB9kfEiJgfb+ixAqeNH8Wj7VsqpSkHIOJfbpUwVf2xM= -
 v1.2.8 h1:hOFUs++dYeZPxWL9rqW0g/ZtfZ+gbghQ/ZOh6q7jCfI= -
 v1.2.9 h1:qol0zAuyJDGW18amWhFCjAkjYnihLWeE8Csnr7JfiYk= -
github.com/docker/swarmkit
 v1.12.0 h1:vcbNXevt9xOod0miQxkp9WZ70IsOCe8geXkmFnXP2e0= h1:n3Z4lIEl7g261ptkGDBcYi/3qBMDl9csaAhwi2MPejs=
github.com/drone/drone
 v0.8.1 h1:idqB65J6E7DRMfV98eN7G6VNHKOLbCq3HTjgWHXfbpo= h1:jhrjj+lR3aY8E5LI+jXxFj4OuXAbdiAIyX0jIwa3OWk=
 v0.8.2 h1:goaJ14Fd24eZR7lCO4gkw9w67PH0hDgYg+CNd0RWI6E= -
 v0.8.3 h1:s/rwzK553WUcjzNu09tO1UM3PdNXxrb+sQw/vOeGtq0= -
 v0.8.4 h1:XfUGVQ72ddm2Z0DDF/2hDZXOFKyS+kNve60DPP4IT4o= -
 v0.8.5 h1:64KbfCmKzxa2Pbq48R8cOXZF5h2NA6yeof544+WUX5w= -
 v0.8.6 h1:3NBcfWTULqVQt28/pRfScRS61Ig7Qtj7b3LGSMWoFDE= -
 v0.8.7 h1:6nNMMXIveNDq89aqhvg1RgCiQmz+EbshGqEmnD6qasw= -
 v0.8.8 h1:tlsAcy+qyp/WG2rp6KOoOGkD0PslGBbA/GIm8Hry8Ck= -
 v0.8.9 h1:Dpl9ewGDujigkHEqtmebCWtVWGo0QzNDSlAmhDnHNT4= -
 v0.8.10 h1:Izqna0P844IYlKp+TrRsSO2BDfCotq3vFtkE9youo+Y= -
github.com/dustin/go-humanize
 v1.0.0 h1:VSnTsYCnlFHaM2/igO1h6X3HA71jcobQuxemgkq4zYo= h1:HtrtbFcZ19U5GC7JDqmcUSB87Iq5E25KnS6fMYU6eOk=
github.com/eapache/go-resiliency
 v1.0.0 h1:XPZo5qMI0LGzIqT9wRq6dPv2vEuo9MWCar1wHY8Kuf4= h1:kFI+JgMyC7bLPUVY133qvEBtVayf5mFgVsvEsIPBvNs=
 v1.1.0 h1:1NtRmCAqadE2FN4ZcN6g90TP3uk8cg9rn9eNK2197aU= -
github.com/eclipse/paho.mqtt.golang
 v0.9.0 h1:7qED2/aWTIBpT72LzH1PkDOqIlwNSq0C7l5cR89s4Gc= h1:H9keYFcgq3Qr5OUJm/JZI/i6U7joQ8SYLhZwfeOo6Ts=
 v0.9.1 h1:d6QTqiGbtCSf8Hz0VCYSZJsXyahZlApicYqoKfgqjog= -
 v1.0.0 h1:SIqh8vC6gWMF+bNWzWh3QUMu+uNlLFkjo2PN9LkJXCE= -
 v1.1.0 h1:Em29HD1CwLHdRFnX7yfg+kBjHHw6DSDok9I+ia4znT4= -
 v1.1.1 h1:iPJYXJLaViCshRTW/PSqImSS6HJ2Rf671WR0bXZ2GIU= -
github.com/edsrzf/mmap-go
 v1.0.0 h1:CEBF7HpRnUCSJgGUb5h1Gm7e3VkmVDrR8lvWVLtrOFw= h1:YO35OhQPt3KJa3ryjFM5Bs14WD66h8eGKpfaBNrHW5M=
github.com/elastic/beats
 v6.4.3+incompatible h1:ZblBvbfOBBpce539Mk4AKgg02DX74t9nVeQR5RyVhxY= h1:7cX7zGsOwJ01FLkZs9Tg5nBdnQi6XB3hYAyWekpKgeY=
 v6.5.0+incompatible h1:AJymh34qIXCcdX12U6hJ0SoSaJ9uYEOLGhN8LJN56dk= -
 v6.5.1+incompatible h1:7/7d+aAyme30L2ri7X3LlYZrxeGQQ8Mxm+vPzEz9BC4= -
 v6.5.2+incompatible h1:KrLUpOtv7PbhSQ0Z36KWmYbv1JocX5RD02AeD2mhQj8= -
 v6.5.3+incompatible h1:TDvV9ARzatDxKl01L44ONWXSr7fnVJ93BtDQIArN5kk= -
 v6.5.4+incompatible h1:pH/LOYaQD+dFFD8D/ZE61jSQtS5tWvMtwNyl4x4M0Ts= -
 v6.6.0+incompatible h1:oqeYovU4JX6EI2d9P3hVEGIidGUYciScqJxaaunJe9o= -
 v7.0.0-alpha1+incompatible h1:Zac+ZSQ1lyLLTgm/gJ2lQqB74xDMLKIM5+NnWVBQMK4= -
 v7.0.0-alpha2+incompatible h1:M26bOoRo2jtyr2ENgryzZdne42wLoXrhq2vWQ6VgNFc= -
 v7.0.0-beta1+incompatible h1:YmWgpJC4Q79/kdJ2gQKtHV7sBjhDxes4PLD5XrJ5v/o= -
github.com/elastic/gosigar
 v0.2.0 h1:GXcDQslFT0IILh6C/G5+rZvP8rMoMLc4+OlG89vtdy8= h1:cdorVVzy1fhmEqmtgqkoE3bYtCfSCkVyjTyCIo22xvs=
 v0.2.1 h1:gsatFyJqe5r+updsdxbh43wLE5moI5aEw99oZ82tB78= -
 v0.3.0 h1:Bw/YCniPJwFUcOp3gL0VFySG9QC6lDEvAinGJx0/sWw= -
 v0.4.0 h1:wGh8Gq7uEIE3uXnz4i6O6W0c6yTsUMF1geDp50YLhno= -
 v0.5.0 h1:mKJ5EXxZ417DGxcBoLd5mDavkeg0bQJw4U6c4D4xLeo= -
 v0.6.0 h1:9KSX3V4Mr15SKb6l6NzTMAYPdoViWQjzCvTRhC9Bgqs= -
 v0.7.0 h1:jct5pZZeiIpv8VWTdFer8jCg7LBhWbsipyCzQOyznC4= -
 v0.8.0 h1:UwJ02o1X/fkWqrEdHbsbjfW3sLfx24/k/QRXqUD7Uz4= -
 v0.9.0 h1:ehdJWCzrtTHhYDmUAO6Zpu+uez4UB/dhH0oJSQ/o1Pk= -
 v0.10.0 h1:bPIzW1Qkut7n9uwvPAXbnLDVEd45TV5ZwxYZAVX/zEQ= -
github.com/elazarl/go-bindata-assetfs
 v1.0.0 h1:G/bYguwHIzWq9ZoyUQqrjTmJbbYn3j3CKKpKinvZLFk= h1:v+YaWX3bdea5J/mo8dSETolEo7R71Vk1u8bnjau5yw4=
github.com/emicklei/go-restful
 v2.2.1+incompatible h1:yreWt49MQDL5ac0Dau9EKE22or+LrHikXVhAqUAXnfk= h1:otzb+WCGbkyDHkqmQmT5YD2WR4BBwUdeQoFo8l/7tVs=
 v2.3.0+incompatible h1:LMJeogUahy5QhG+/aiNAz8ZSA/2pSPzlwMpiaxRDMOM= -
 v2.4.0+incompatible h1:p9u+CKd2OEI+kUmFLDwuf0LtmBtDhcok4UjQDs0rDDk= -
 v2.5.0+incompatible h1:C6LOcwNPrNImeYfAr02vGeM0Mpd7mE2CqSR2mpPd4p4= -
 v2.6.0+incompatible h1:luAX89wpjId5gV+GJV11MFD56GpAJTG2eUqCeDDgB98= -
 v2.6.1+incompatible h1:wgryk4OPtwC4O5lLa33ldKjliZhL/iVZHK8egL/p5rM= -
 v2.7.0+incompatible h1:DLOt75KPGt7LnFiqlmQGKImiR+updEs3F5/wrYO9P5k= -
 v2.8.0+incompatible h1:wN8GCRDPGHguIynsnBartv5GUgGUg1LAU7+xnSn1j7Q= -
 v2.8.1+incompatible h1:AyDqLHbJ1quqbWr/OWDw+PlIP8ZFoTmYrGYaxzrLbNg= -
 v2.9.0+incompatible h1:YKhDcF/NL19iSAQcyCATL1MkFXCzxfdaTiuJKr18Ank= -
github.com/emirpasic/gods
 v1.5.1 h1:rL1Ty4inU7bNBfAJtKfFPkH6hrNql//WDd2E/OC6yjI= h1:YfzfFFoVP/catgzJb4IKIqXjX78Ha8FMSDh3ymbK86o=
 v1.5.2 h1:tXv4AzGxjsFEHAcuh0XidoM7nGgqAoKH2mkT5X7qKJI= -
 v1.6.0 h1:19k4VMJEPwdKPi7YOmv8Y+BdVpH4vqdPFm2ATm1w6x8= -
 v1.7.0 h1:rhvbuJ3/m4qDAc2eJYrDpkzbRx5UbnWuz0t5erFC2oc= -
 v1.8.0 h1:YHvS59P0319zAckH7tphI0U7h+3fdxaMelCpzMFYZqo= -
 v1.8.1 h1:PnIAh9bXmZhRh4UaoNXB+UtJIOCIRYvWx8k/h+V4ilU= -
 v1.9.0 h1:rUF4PuzEjMChMiNsVjdI+SyLu7rEqpQ5reNFnhC7oFo= -
 v1.10.0 h1:bWdjTVTw8eMGzGzvpg03dqalNduWpWVfleguQlxt9+0= -
 v1.11.0 h1:8KhjokrJy1+REZkLeSlnJvLKI4tRR8g65a4C07oQQ50= -
 v1.12.0 h1:QAUIPSaCu4G+POclxeqb3F+WPpdKqFGlw36+yOzGlrg= -
github.com/ethereum/go-ethereum
 v1.8.14 h1:q+r1V2aNLAlE3+5FsRM5d2hQF4Wc7h7PfhSeSoq6zUo= h1:PwpWDrCLZrV+tfrhqqF6kPknbISMHaJv9Ln3kPCZLwY=
 v1.8.15 h1:95jix4Qx65CpTBLjzrh8Jb8UR/3TaolQE82qFQZCJJY= -
 v1.8.16 h1:mWSfeuH5G3WDvHcY/aXDgSjy+mXQ6UyL7w0hrMbE/ds= -
 v1.8.17 h1:aoqWfGFYsSxCdFZfQ6h0pnojtoBOcYI+6Yg8JXhGuXs= -
 v1.8.18 h1:ihTGGzBE6umIZsjR7Iijvlx0Ri0TqazhxRvdkIyBBok= -
 v1.8.19 h1:H3m/wLo1hx3gpduVMQWb9ertqkmuTXvqjQNcxiA6XNI= -
 v1.8.20 h1:Sr6DLbdc7Fl2IMDC0sjF2wO1jTO5nALFC1SoQnyAQEk= -
 v1.8.21 h1:ofzsxFj+zKhj1k3uVa8/MJCCptqKEh/6DXq4LwdUM4E= -
 v1.8.22 h1:y8RPBpBOF0/Gm8tV4Ut0WMa6RvY0e4XFIT6zASAOT0I= -
 v1.8.23 h1:xVKYpRpe3cbkaWN8gsRgStsyTvz3s82PcQsbEofjhEQ= -
github.com/evanphx/json-patch
 v3.0.0+incompatible h1:l91aby7TzBXBdmF8heZqjskeH9f3g7ZOL8/sSe+vTlU= h1:50XU6AFN0ol/bzJsmQLiYLvXMP4fmwYFNcr97nuDLSk=
 v4.0.0+incompatible h1:xregGRMLBeuRcwiOTHRCsPPuzCQlqhxUPbqdw+zNkLc= -
 v4.1.0+incompatible h1:K1MDoo4AZ4wU0GIU/fPmtZg7VpzLjCxu+UwBD1FvwOc= -
github.com/fatih/camelcase
 v1.0.0 h1:hxNvNX/xYBp0ovncs8WyWZrOrpBNub/JfaMvbURyft8= h1:yN2Sb0lFhZJUdVvtELVWefmrXpuZESvPmqwoZc+/fpc=
github.com/fatih/color
 v0.2.0 h1:Xr6X9g4H4m79h2IHRx6o2V0NFOUbGwQZdptjcetM9nA= h1:Zm6kSWBoL9eyXnKyktHP6abPY2pDugNf5KwzbycvMj4=
 v1.0.0 h1:4zdNjpoprR9fed2QRCPb2VTPU4UFXEtJc9Vc+sgXkaQ= -
 v1.1.0 h1:4RQHlUrrLRssqNPpcM+ZLy+alwucmC4mkIGTbiVdCeY= -
 v1.3.0 h1:YehCCcyeQ6Km0D6+IapqPinWBK6y+0eB5umvZXK9WPs= -
 v1.4.0 h1:zw+qNRz7futKZsgbBmT5ffigcBJLFNu0TZAxbTxFr8U= -
 v1.4.1 h1:YJhD/SoQqn7ev9zwhIm7lHTAqsOAF2AN4xlAVZzNZnU= -
 v1.5.0 h1:vBh+kQp8lg9XPr56u1CPrWjFXtdphMoGWVHr9/1c+A0= -
 v1.6.0 h1:66qjqZk8kalYAvDRtM1AdAJQI0tj4Wrue3Eq3B3pmFU= -
 v1.7.0 h1:DkWD4oS2D8LGGgTQ6IvwJJXSL5Vp2ffcQg58nFV38Ys= -
github.com/fatih/structs
 v1.0.0 h1:BrX964Rv5uQ3wwS+KRUAJCBBw5PQmgJfJ6v4yly5QwU= h1:9NiDSp5zOcgEDl+j00MP/WkGVPOlPRLejGD8Ga6PJ7M=
 v1.1.0 h1:Q7juDM0QtcnhCpeyLGQKyg4TOIghuNXrkL32pHAUMxo= -
github.com/fluent/fluent-logger-golang
 v0.4.3 h1:wj4AQt4A4po5vyZ8VaYR0muRsbiYJECiL+y8KCvt/To= h1:2/HCT/jTy78yGyeNGQLGQsjF3zzzAuy6Xlk6FCMV5eU=
 v0.4.4 h1:f+o4uPhTfeRZ+TjuB/vgvV57UV1tnPle/p7AGO+cQzo= -
 v0.5.0 h1:WH9AKTauVdti3T9xu0fol/oTUzXsnmHpYHi3bqLU6ug= -
 v0.5.1 h1:NEM0cyfUc51PRtZgPqRbN7R3YabR9v3oA2loOseoP34= -
 v1.0.0 h1:hkXcdtGofeByowrIO47wiK2ETGq43UG7VZWWLO+Y3Eg= -
 v1.1.0 h1:etYzQE058bvZu6Iy8MRft9eW0Xq4AxwkEZGlVPNeeEk= -
 v1.2.0 h1:FmTE6OWpD+sFivPK91k6yeh2Oia/T0wImmpY5LkGNjA= -
 v1.2.1 h1:CMA+mw2zMiOGEOarZtaqM3GBWT1IVLNncNi0nKELtmU= -
 v1.3.0 h1:oBolFKS9fY9HReChzaX1RQF5GkdNdByrledPTfUWoGA= -
 v1.4.0 h1:uT1Lzz5yFV16YvDwWbjX6s3AYngnJz8byTCsMTIS0tU= -
github.com/fluffle/goirc
 v1.0.0 h1:4ejJt3nRw65AXzSYmHbMWYrlPHqc+ycoJQ0saZ/F4ZY= h1:XSqSOq5nTMabnZQgLQMKdkleBiUntueKLNTb0PpSOZw=
 v1.0.1 h1:YHBfWIXSFgABz8dbijvOIKucFejnbHdk78+r2z/6B/Q= h1:bm91JNJ5r070PbWm8uG9UDcy9GJxvB6fmVuHDttWwR4=
github.com/fsnotify/fsnotify
 v1.2.8 h1:izx5kyspXf0br68OpIhHsLlY2e27kdAdP44+UOBxPoM= h1:jwhsz4b93w/PPRr/qN1Yymfu8t87LnFCMoQvtojpjFo=
 v1.2.9 h1:q+f2SvddJCJUSZ2be2WshDfu4y08zDzn3EgNWoaU9Nw= -
 v1.2.10 h1:rtNmYN+rpP0715msYFFD9UXCWMdxXqXub9G4RdR/Kck= -
 v1.2.11 h1:yvKD8fnPUn9IDsfDnesbiAnEaI5WMA9/sUxIy6vltDc= -
 v1.3.0 h1:XyNoRE4PlEAzjaHYBqJuJC18jNWyfDVJ2jwD5kCuwXs= -
 v1.3.1 h1:Ls7eCFzutKKUlbY8E5wXfcRWGH4qLQxkDoydIPh94X4= -
 v1.4.0 h1:WphdbukyYaFeCBwYjXl+GdYBLBrC754TP7DNLI8Qyfs= -
 v1.4.1 h1:eFkOCdozfhibgldtd1nfPgoQJDlDwJt4n6DRqtIZ0z0= -
 v1.4.2 h1:v5tKwtf2hNhBV24eNYfQ5UmvFOGlOCmRqk7/P1olxtk= -
 v1.4.7 h1:IXs+QLmnXW2CcXuY+8Mzv/fWEsPGWxqefPtCP5CnV9I= -
github.com/fsouza/go-dockerclient
 v1.2.0 h1:lmjnUbZFR+De5PfuXCk62hAIlkokNtD9CaGPR8Iu5UU= h1:KpcjM623fQYE9MZiTGzKhjfxXAV9wbyX2C1cyRHfhl0=
 v1.2.1 h1:ZcSDAjMR2wkfuAOOaoCOML8NZKuXRi8L0aib5ZtGPoc= -
 v1.2.2 h1:rFDrkgZUIlruULXD2gRhT8JhqbjA6vHszAIStg/juEY= -
 v1.3.0 h1:tOXkq/5++XihrAvH5YNwCTdPeQg3XVcC6WI2FVy4ZS0= h1:IN9UPc4/w7cXiARH2Yg99XxUHbAM+6rAi9hzBVbkWRU=
 v1.3.1 h1:h0SaeiAGihssk+aZeKohbubHYKroCBlC7uuUyNhORI4= -
 v1.3.2 h1:FrEqffNYbNeH35BB+UdeJszz53nMqADuXsNAUt7m11o= h1:yKgNDnynLAmC24gq8gnW2Yu3QQMeMz1tWYdWWl1S04w=
 v1.3.3 h1:HaGPJlgBFwCy3W7K/FD5IobA7uuy0RPcAcfa3azv+b4= h1:wqOJeWHV3Mep81Dx04uGm0ovCSZxtubD4at6XGfEPh0=
 v1.3.4 h1:X5OplZ137gPE1dzOTFsy7sxDiz73JVh6JZwjQovFka4= h1:6VxxpLZHol+SXg45PuAgkYwQCjo5KL0A7VoWkQOWAqU=
 v1.3.5 h1:0/nSVi1SRCW7VQ2kRLn56Uw2YybNRdY/iGrnWjFycXI= -
 v1.3.6 h1:oL0e3fpCjF+AHuUUBnwbkVcelFhxQifgTPQKipJPtnI= h1:ptN6nXBwrXuiHAz2TYGOFCBB1aKGr371sGjMFdJEr1A=
github.com/fzzy/radix
 v0.4.6 h1:WT0tffvXEMvi+WE1i8b0IBi4Oh/8Yqt4qBCPgjfEoF8= h1:KhtJfdbo4PD2LEOYO7QCVSIH0pOcZEZ/SpNsXgwQtkk=
 v0.4.7 h1:UUneQwfdGF5n8G9ZpOCTavs2cUUY1itsjfBPT/BNAow= -
 v0.4.8 h1:/Boc+Pf4DpvhBdSMtkAlMHtdz/bZSDJRAKe8V+956Mk= -
 v0.5.0 h1:gHh2VDUKGMPQaNGF9VIfcaHhPIcdzl8Q4XuDbtmcD/4= -
 v0.5.1 h1:MkuP8PECSAZQva1Vdh6DvS6gYr6A3Phrd2jKZJgLdSw= -
 v0.5.2 h1:wfFq00/ff8+SSw8IDen7MS0q9k3yAsmWSdSXDvzJ+Tw= -
 v0.5.3 h1:T9TZ6MaNkR8BvqyjvmYnxOTtbleSdHCyimHEd9+Aep0= -
 v0.5.4 h1:1fLdjfVd/eM6QK8d+di+njsoUFhgyf021kDqjXBlLQU= -
 v0.5.5 h1:GYvT/gqJBruVcBBybBBWvcL9miq5MxwNqFoZmRsITRM= -
 v0.5.6 h1:cbj4zksFVtUo5ST6gW5NsUlW6C6x7eAiqajgCe6e2Ys= -
github.com/gambol99/go-marathon
 v0.1.0 h1:lK8p8xRbeTyMzjtBXsTp8Wo3zlcEO9NNLjii/VF/wTQ= h1:GLyXJD41gBO/NPKVPGQbhyyC06eugGy15QEZyUkE2/s=
 v0.1.1 h1:bvvsPzbxJ3cYxIE7kGOkG5DeQxJn4UmCQWMwwEl04eE= -
 v0.2.0 h1:A/THKKbD31WtqSyHCIv+KseL5Ovl7E0PuR90dH4ZxuI= -
 v0.3.0 h1:PRTm+aaTMFDR94wzEhjkzGndbrVaraUdGYkDLdcjwv4= -
 v0.4.0 h1:wvV2zI++OmhFkdmiZYtvVssIqTeKXDowoqA0yE+vkvU= -
 v0.5.0 h1:hS239GBgu76JGsHc4bproQuj4dhLgPtMZlTyoRAekHs= -
 v0.5.1 h1:9zeoAC/8n7aWScFgeY5u9miLuKPbGm6xLvVI3n5GSIc= -
 v0.6.0 h1:uxJ3FduoL67JwV+O4FhUmQs1QK/fJZ0HOXXQT6UBWas= -
 v0.7.0 h1:T7klmdocOKUtshucgKrqipH1YA71AGXETL6DX8jjdZA= -
 v0.7.1 h1:/dnwXQ0W0UDScpvmcdjzRz3ssnJ/5ieX/q4Xi/QHOn4= -
github.com/garyburd/redigo
 v1.0.0 h1:W6d6zr96WMrMxQws1I4sc7rrJ1dbQK5KrC+NwH0ReTM= h1:NR3MbYisc3/PwhQ00EMzDiPmrwpPxAn5GI05/YaO1SY=
 v1.1.0 h1:kTY6M1SUxdOiFU4rbXWTtDBsTnfsXo4vDhXzhGMjdwk= -
 v1.2.0 h1:l2RL1LG+FC1gM5pm6nZUtWBYuALPJ+Bc6qEgWdCVeYQ= -
 v1.3.0 h1:gjl0wbI1VZoOZvwJge1tGXZX8rdbwo91iVRPV13wDu0= -
 v1.4.0 h1:PlMIyh8f7og1DRVAZiU1I6VR8R5vFbWch3ddfv1ICvY= -
 v1.5.0 h1:OcZhiwwjKtBe7TO4TlXpj/1E3I2RVg1uLxwMT4VFF5w= -
 v1.6.0 h1:0VruCpn7yAIIu7pWVClQC8wxCJEcG3nyzpMSHKi1PQc= -
github.com/gdamore/tcell
 v1.0.0 h1:oaly4AkxvDT5ffKHV/n4L8iy6FxG2QkAVl0M6cjryuE= h1:tqyG50u7+Ctv1w5VX67kLzKcj9YXR/JSBZQq/+mLl1A=
 v1.1.0 h1:RbQgl7jukmdqROeNcKps7R2YfDCQbWkOd1BwdXrxfr4= -
 v1.1.1 h1:U73YL+jMem2XfhvaIUfPO6MpJawaG92B2funXVb9qLs= h1:K1udHkiR3cOtlpKG5tZPD5XxrF7v2y7lDq7Whcj+xkQ=
github.com/gengo/grpc-gateway
 v1.4.0 h1:dFHVf5i6z1HqoLVTZxKwuMccH+slBVN7sRY3pzFgTcE= h1:96Q3MwP4ORaK7X4PLNhexJ67u+39FMqYtFT13kAdQcU=
 v1.4.1 h1:T5yx4HlW1/vwKV84f2UZOv4nGrF1rZTgC7ULSbFEhCQ= -
 v1.5.0 h1:nB/ochpeZ16sWKWYh2TTnxzNJwrFlAH5VCSQVQp/4Zk= -
 v1.5.1 h1:8oHtTCae6X65dEoplT9NXM55eJr3CSrzMGSWzcWd9Hs= -
 v1.6.0 h1:cJWEMz8E5vNZkqMjUTuRl3llRT5ao+WCwF4RCk/Jjus= -
 v1.6.1 h1:04qlHOgsm5e6ExdBbgMlp/uDl8HgNAPkUucE/yacu68= -
 v1.6.2 h1:iz2cj8mbdvToKMeUIpJZ5Ma38fyWjb5uhMQ4fPHnigg= -
 v1.6.3 h1:YX3k6oQBxABuBQNzM0DcnAzKOMRPR4plpcAO+VHy9Z0= -
 v1.6.4 h1:ERYqE5SsnCwlE6WjFa22lPcQQWriDGvb3mB3uBKJrLs= -
 v1.7.0 h1:RdBowFlRqkvwBlQxwg/DqDRY5eINKNIfeheMpDhAxZg= -
github.com/getsentry/raven-go
 v0.1.0 h1:lc5jnN9D+q3panDpihwShgaOVvP6esoMEKbID2yhLoQ= h1:KungGk8q33+aIAZUIVWZDr2OfAEBsO49PX4NzFV5kcQ=
 v0.1.1 h1:Q59NpKCRMFKdZWk2H5mzc5N9IEe+Hx/kZKd8w4dCvs4= -
 v0.1.2 h1:4V0z512S5mZXiBvmW2RbuZBSIY1sEdMNsPjpx2zwtSE= -
 v0.2.0 h1:no+xWJRb5ZI7eE8TWgIq1jLulQiIoLG0IfYxv5JYMGs= -
github.com/ghodss/yaml
 v1.0.0 h1:wQHKEahhL6wmXdzwWG11gIVCkOv05bNOh+Rxn0yngAk= h1:4dBDuWmgqj2HViK6kFavaiC9ZROes6MMH2rRYeMEF04=
github.com/gin-gonic/gin
 v1.1.1 h1:RsFHhz6Sl1rRi1bjR0zrynA9WUMxOHywfoDP1+UpoJE= h1:7cKuhb5qV2ggCFctp2fJQ+ErvciLZrIeoOSOm6mUr7Y=
 v1.1.2 h1:F26LJETbvvzxGkpOWRHyOXF/s3sdNnv4EIVbBMMDrXo= -
 v1.1.3 h1:A2cKDURjNpMkEJ7ts+gD+T/jVHeo+fgDB1aRIUXAo5U= -
 v1.1.4 h1:XLaCFbU39SSGRQrEeP7Z7mM3lvRqC4vE5tEaVdLDdSE= -
 v1.3.0 h1:kCmZyPklC0gVdL728E6Aj20uYBJV93nj/TkwBTKhFbs= -
github.com/gizak/termui
 v2.1.1+incompatible h1:pvS3JLIPOBD/fTHn8po0eImr5QsoaIIuJ3Fnn34zBiA= h1:PkJoWUt/zacQKysNfQtcw1RW+eK2SxkieVBtl+4ovLA=
 v2.2.0+incompatible h1:qvZU9Xll/Xd/Xr/YO+HfBKXhy8a8/94ao6vV9DSXzUE= -
 v2.3.0+incompatible h1:S8wJoNumYfc/rR5UezUM4HsPEo3RJh0LKdiuDWQpjqw= -
github.com/gliderlabs/logspout
 v3.2.1+incompatible h1:qVUKq8Yc+QuxpUziJ0l/vT2+Vh9YZIwDihVLdcTf9Is= h1:fBo/Nq8zbRI0AUwYxcIDPEOG/J4reIVOAHUbmohj2yk=
 v3.2.2+incompatible h1:rSDcVw4uiCZ2zc/0yOw5JP/98gzvW3hwiHJ3cQvKqB8= -
 v3.2.3+incompatible h1:aogICzHq/SUblF3Y5/V7oI93IW3LGjsZdcRpCn3v9oo= -
 v3.2.4+incompatible h1:K2ijThhXN7DzUCULuE3yRGbWAcSt1Xji/Sv1v0G/XZM= -
 v3.2.5+incompatible h1:fuWjCJJcIPG6JlKwSHg8d56l8pSUXIsOysZOh1dXIvA= -
 v3.2.6+incompatible h1:/9k8CrainVIA+kmP8GHREWRo4ug7+Fx45A7+0jikTs4= -
github.com/go-chi/chi
 v3.2.0+incompatible h1:J6iWCmCXnsUtK7kBKiY+YS2Oq5+LJJ0g71bUjSewvXk= h1:eB3wogJHnLi3x/kFX2A+IbTBlXxmMeXJVKy9tTv1XzQ=
 v3.2.1+incompatible h1:f/Wdc+ueut4zFR+OXY2zBjlLR7FiCSCPia7W0YHI60o= -
 v3.3.0+incompatible h1:19pl0NEHtjUmuCdXZpZ4RP3dJWdf05Fg8DDTFLnq++8= -
 v3.3.1+incompatible h1:ib+xvnkWGS9Dt+qDnW03Sr7j2vrUjaTu3hDiR4Phd+Q= -
 v3.3.2+incompatible h1:uQNcQN3NsV1j4ANsPh42P4ew4t6rnRbJb8frvpp31qQ= -
 v3.3.3+incompatible h1:KHkmBEMNkwKuK4FdQL7N2wOeB9jnIx7jR5wsuSBEFI8= -
 v3.3.4+incompatible h1:X+OApYAmoQS6jr1WoUgW+t5Ry5RYGXq2A//WAL5xdAU= -
 v4.0.0-rc2+incompatible h1:A7EBPJSnKP0u77IKYE/fS8s4zngJirbJvxIDl37bOn0= -
 v4.0.0+incompatible h1:SiLLEDyAkqNnw+T/uDTf3aFB9T4FTrwMpuYrgaRcnW4= -
 v4.0.1+incompatible h1:RSRC5qmFPtO90t7pTL0DBMNpZFsb/sHF3RXVlDgFisA= -
github.com/go-errors/errors
 v1.0.0 h1:2G1gYpeHw4GhLet4Ebp5q9wpnSCAOJNTiJq+I3wJV5I= h1:f4zRHt4oKfwPJE5k8C9vpYG+aDHdBFUsgrm6/TyX73Q=
 v1.0.1 h1:LUHzmkK3GUKUrL/1gfBUxAHzcev3apQlezX/+O7ma6w= -
github.com/go-gorp/gorp
 v1.2.1 h1:cyvKaN0GZxHM23au9yjuMKiBDarCbXWkpVZsOR4M8Lw= h1:7IfkAQnO7jfT/9IQ3R9wL1dFhukN6aQxzKTHnkxzA/E=
 v1.6.1 h1:gmMbEEupmbMu7tPNZic1tNmTrCpnbfrdeaqisMto7lk= -
 v1.7.1 h1:iah1rpKL35iFU2EFgJ595LG03h6jrbl1WIEWvPmzjo4= -
 v1.7.2 h1:C5uGH8zK2qjMJZGC308ZegdGXMrMjYmA++IIMeKSKnc= -
 v2.0.0+incompatible h1:dIQPsBtl6/H1MjVseWuWPXa7ET4p6Dve4j3Hg+UjqYw= -
github.com/go-ini/ini
 v1.38.1 h1:hbtfM8emWUVo9GnXSloXYyFbXxZ+tG6sbepSStoe1FY= h1:ByCAeIL28uOIIG0E3PJtZPDL8WnHpFKFOtgjp+3Ies8=
 v1.38.2 h1:6Hl/z3p3iFkA0dlDfzYxuFuUGD+kaweypF6btsR2/Q4= -
 v1.38.3 h1:CclkQtfmOJadMVMYepq1DkVSYw2jf/0BTvjNBHth5xY= -
 v1.39.0 h1:/CyW/jTlZLjuzy52jc1XnhJm6IUKEuunpJFpecywNeI= -
 v1.39.1 h1:tCmZ4eaQ/68aQjBmdycPGDhhfHsu8f/O3kM/nt6JZGg= -
 v1.39.2 h1:mznOicgW6rGbX0ZaiSfOgrYoEq+H/bHUJsTfEBbGhWI= -
 v1.39.3 h1:y2UyknTfDmqZcBqdAHMt3zib4YT33TVtM6ABVrRVXQ0= -
 v1.40.0 h1:/pbZah2UXAjMCtUlVRASCb6nX+0A8aCXjmYouBEXu0c= -
 v1.41.0 h1:526aoxDtxRHFQKMZfcX2OG9oOI8TJ5yPLM0Mkno/uTY= -
 v1.42.0 h1:TWr1wGj35+UiWHlBA8er89seFXxzwFn11spilrrj+38= -
github.com/go-kit/kit
 v0.1.0 h1:M3u0OQMyt/4O3npZa1iK0/zGSKTfSvPDDHcG1LAnY8w= h1:xBxKIO96dXMWWy0MnWVtmwkA9/13aqxPnvrjFYMA2as=
 v0.2.0 h1:96r99Qs3P4+3BJ9DTG79dapahoHO/0+q9v0xsifBFck= -
 v0.3.0 h1:QZEva+odUF/G+yz7yjQLwUQxnSAS4S45V9+4O02yJ1Q= -
 v0.4.0 h1:KeVK+Emj3c3S4eRztFuzbFYb2BAgf2jmwDwyXEri7Lo= -
 v0.5.0 h1:SI25KgiIaNiy8GCcvstnkBVXPISD0rJ7LrAwt1PJ8zA= -
 v0.6.0 h1:wTifptAGIyIuir4bRyN4h7+kAa2a4eepLYVmRe5qqQ8= -
 v0.7.0 h1:ApufNmWF1H6/wUbAG81hZOHmqwd0zRf8mNfLjYj/064= -
 v0.8.0 h1:Wz+5lgoB0kkuqLEc6NVmwRknTKP6dTGbSqvhZtBI/j0= -
github.com/go-ldap/ldap
 v2.2.1+incompatible h1:xiQNn/mDyO2c0O/3+dJJBn3M63u0eJV0AWyXbW4DSEg= h1:qfd9rJvER9Q0/D/Sqn1DfHRoBp40uXYvFoEVrNEPqRc=
 v2.2.2+incompatible h1:z1iMOfXVEARpVjU9RFBoHGERz/ZZlNsq/eLyzzNll/g= -
 v2.3.0+incompatible h1:qpbRTQISA20SoOhwzpkj4DuWDbsv9Z0Rgc88hpSPwvg= -
 v2.4.0+incompatible h1:272V7fzoTx3z7o75WQSHL6bjm8sJuMa2oUQzh2S1xEY= -
 v2.4.1+incompatible h1:MuJc8IOP4tcuWQreeSoiLD/GXZhV5vJaAbL7o17Bqug= -
 v2.5.0+incompatible h1:q4gje9ELj+aiS1Y120Mm8J533cwz70m6qMdDo86Ztng= -
 v2.5.1+incompatible h1:Opaoft5zMW8IU/VRULB0eGMBQ9P5buRvCW6sFTRmMn8= -
 v3.0.0+incompatible h1:Q5Cni6EaH1FBILu1vuwwPd3Bosl3XYEdSoDKgyO/Uis= -
 v3.0.1+incompatible h1:HZ4m1DxAjmIXb/0JzRu9YXxdlEIhzfhQnKtSsjMgoUE= -
github.com/go-logfmt/logfmt
 v0.1.0 h1:7paswH2J8PJSBWgCVBz4+eagiCOnMzVJ1CGUd414+MY= h1:Qt1PoO58o5twSAckw1HlFXLmHsOX5/0LbT9GBnD5lWE=
 v0.2.0 h1:2e4QP7mYUCi0P4yP/sfHF6unrAXvTCLLaU5tFV9oWJE= -
 v0.3.0 h1:8HUsc87TaSWLKwrnumgC8/YconD2fJQsRJAsWaPg2ic= -
 v0.4.0 h1:MP4Eh7ZCb31lleYCFuwm0oe4/YGak+5l1vA2NOE80nA= h1:3RMwSq7FuexP4Kalkev3ejPJsZTpXXBr9+V4qmtdjCk=
github.com/go-openapi/errors
 v0.17.0 h1:g5DzIh94VpuR/dd6Ff8KqyHNnw7yBa2xSHIPPzjRDUo= h1:LcZQpmvG4wyF5j4IhA73wkLFQg+QJXOQHVjmcZxhka0=
 v0.17.1 h1:5Fq3wlwS3oF+a3ogdmAovUBiGFa2cvL88gK++KzzkpA= -
 v0.17.2 h1:azEQ8Fnx0jmtFF2fxsnmd6I0x6rsweUF63qqSO1NmKk= -
 v0.18.0 h1:+RnmJ5MQccF7jwWAoMzwOpzJEspZ18ZIWfg9Z2eiXq8= -
github.com/go-openapi/loads
 v0.17.0 h1:H22nMs3GDQk4SwAaFQ+jLNw+0xoFeCueawhZlv8MBYs= h1:72tmFy5wsWx89uEVddd0RjRWPZm92WRLhf7AC+0+OOU=
 v0.17.1 h1:yTcHtvNp8szdZ6UXm4h+c+xU4Qp30NWMYNP39cHnWmQ= -
 v0.17.2 h1:tEXYu6Xc0pevpzzQx5ghrMN9F7IVpN/+u4iD3rkYE5o= -
 v0.18.0 h1:2A3goxrC4KuN8ZrMKHCqAAugtq6A6WfXVfOIKUbZ4n0= -
github.com/go-openapi/runtime
 v0.17.0 h1:NUysn+2kDjI+GbS5usELZM8bfOyntKKOGY4uwDEeCq0= h1:QO936ZXeisByFmZEO1IS1Dqhtf4QV1sYYFtIq6Ld86Q=
 v0.17.1 h1:STQHpGAn63Ij0sI57fEHKIvtBI3v+RBozFJuEOE1Ps4= -
 v0.17.2 h1:/ZK67ikFhQAMFFH/aPu2MaGH7QjP4wHBvHYOVIzDAw0= -
 v0.18.0 h1:ddoL4Uo/729XbNAS9UIsG7Oqa8R8l2edBe6Pq/i8AHM= h1:uI6pHuxWYTy94zZxgcwJkUWa9wbIlhteGfloI10GD4U=
github.com/go-openapi/spec
 v0.17.0 h1:XNvrt8FlSVP8T1WuhbAFF6QDhJc0zsoWzX4wXARhhpE= h1:XkF/MOi14NmjsfZ8VtAKf8pIlbZzyoTvZsdfssdxcBI=
 v0.17.1 h1:ZfZ1w1bBW6QC7EXXz3DXQDlFLCaPo4Nhszx5uOTjT2Q= -
 v0.17.2 h1:eb2NbuCnoe8cWAxhtK6CfMWUYmiFEZJ9Hx3Z2WRwJ5M= -
 v0.18.0 h1:aIjeyG5mo5/FrvDkpKKEGZPmF9MPHahS72mzfVqeQXQ= -
github.com/go-openapi/strfmt
 v0.17.0 h1:1isAxYf//QDTnVzbLAMrUK++0k1EjeLJU/gTOR0o3Mc= h1:P82hnJI0CXkErkXi8IKjPbNBM6lV6+5pLP5l494TcyU=
 v0.17.1 h1:o/yBocNZGzjYbJYu6ApCD9SWj8WRNWtg2apipkZEtk8= -
 v0.17.2 h1:2KDns36DMHXG9/iYkOjiX+/8fKK9GCU5ELZ+J6qcRVA= -
 v0.18.0 h1:FqqmmVCKn3di+ilU/+1m957T1CnMz3IteVUcV3aGXWA= -
github.com/go-openapi/swag
 v0.17.0 h1:iqrgMg7Q7SvtbWLlltPrkMs0UBJI6oTSs79JFRUi880= h1:AByQ+nYG6gQg71GINrmuDXCPWdL640yX49/kXLo40Tg=
 v0.17.1 h1:05rL2ATPnpCFQxLDBrCQ91n/bJxkxKRfghvuk+d6fLI= -
 v0.17.2 h1:K/ycE/XTUDFltNHSO32cGRUhrVGJD64o8WgAIZNyc3k= -
 v0.18.0 h1:1DU8Km1MRGv9Pj7BNLmkA+umwTStwDHttXvx3NhJA70= -
github.com/go-openapi/validate
 v0.17.0 h1:pqoViQz3YLOGIhAmD0N4Lt6pa/3Gnj3ymKqQwq8iS6U= h1:Uh4HdOzKt19xGIGm1qHf/ofbX1YQ4Y+MYsct2VUrAJ4=
 v0.17.1 h1:RfQTLHm/gEu0oSUmbTOy0PMufjkE5/pPfnqYpor3WLc= -
 v0.17.2 h1:lwFfiS4sv5DvOrsYDsYq4N7UU8ghXiYtPJ+VcQnC3Xg= -
 v0.18.0 h1:PVXYcP1GkTl+XIAJnyJxOmK6CSG5Q1UcvoCvNO++5Kg= -
github.com/go-redis/redis
 v6.11.0+incompatible h1:HVwqrD0lHOxaZ/S6T8ScWo8JS4UHnZxMqg+LPEVKWxo= h1:NAIEuMOZ/fxfXJIrKDQDz8wamY7mA7PouImQ2Jvg6kA=
 v6.12.0+incompatible h1:s+64XI+z/RXqGHz2fQSgRJOEwqqSXeX3dliF7iVkMbE= -
 v6.13.0+incompatible h1:ogn3rdRIVfT9NmMdLgB+B3MoHbsm9JsYmvYbwCi5IgM= -
 v6.13.1+incompatible h1:7EyNTuE9zwllLsn73pNTUwVTlrgs4vgNWZ9yx1nCTvQ= -
 v6.13.2+incompatible h1:kfEWSpgBs4XmuzGg7nYPqhQejjzU9eKdIL0PmE2TtRY= -
 v6.14.0+incompatible h1:AMPZkM7PbsJbilelrJUAyC4xQbGROTOLSuDd7fnMXCI= -
 v6.14.1+incompatible h1:kSJohAREGMr344uMa8PzuIg5OU6ylCbyDkWkkNOfEik= -
 v6.14.2+incompatible h1:UE9pLhzmWf+xHNmZsoccjXosPicuiNaInPgym8nzfg0= -
 v6.15.0+incompatible h1:/Wib9cA7CF3SQxBZRMHyQvqzlwzc8PJGDMkRfqQebSE= -
 v6.15.1+incompatible h1:BZ9s4/vHrIqwOb0OPtTQ5uABxETJ3NRuUNoSUurnkew= -
github.com/go-sql-driver/mysql
 v1.0.0 h1:UhERwrakx3lDloHVJA0uKtkiCVovXzcGLwz6XuNt6ks= h1:zAC/RDZ24gD3HViQzih4MyKcchzm+sOG5ZlKdlhCg5w=
 v1.0.1 h1:3V2p1zqhZ2bY30jTD3lLL6ipcE9ARatt/P+u6Z38FLE= -
 v1.0.2 h1:1tKe25AbTHPop81Rx2y5qnt3L7p+2fvuf4Qc9ZSOt2A= -
 v1.0.3 h1:rpz2dLylQZCdCT+t3aBXK1IFdfv1ttTkuFqaAYoy2BA= -
 v1.1.0 h1:9+YfHL3eyxobwWIChLZyZ20UeNW5HM8/IOcl3OWBOpk= -
 v1.2.0 h1:C5cl8DzJiobQuZhND5+a3cOrrRhyaJBPHxZjLgdN8kk= -
 v1.3.0 h1:pgwjLi/dvffoP9aabwkT3AKpXQM93QARkjFhDDqC1UE= -
 v1.4.0 h1:7LxgVwFb2hIQtMm87NdgAVfXjnt4OePseqT1tKx+opk= -
 v1.4.1 h1:g24URVg0OFbNUTx9qqY1IRZ9D9z3iPyi5zKhQZpNwpA= -
github.com/go-stack/stack
 v1.4.0 h1:OT8w0Jz6Dzuc6zZd8ExJJ+8B+sgrmUHUogPAPm3XsJ0= h1:v0f6uXyyMGvRgIKkXu+yp6POWl0qKG85gN/melR3HDY=
 v1.5.0 h1:NnuewVnme84IpkOVfD33YeVxeYCEadBkpqvLtp4ze3M= -
 v1.5.1 h1:LIG9n/P1KDjrzWjeMtiau4ltqU4NdHn4PCluFXMubRY= -
 v1.5.2 h1:5sTB/0oZM2O31k/N1IRwxxVXzLIt5NF2Aqx/2gWI9OY= -
 v1.5.3 h1:JxU+OTZ+vcOqLFrCTritv2agi57iNoCkA14tYsEVfNs= -
 v1.5.4 h1:ACUuwAbOuCKT3mK+Az9UrqaSheA8lDWOfm0+ZT62NHY= -
 v1.6.0 h1:MmJCxYVKTJ0SplGKqFVX3SBnmaUhODHZrrFF6jMbpZk= -
 v1.7.0 h1:S04+lLfST9FvL8dl4R31wVUC/paZp/WQZbLmUgWboGw= -
 v1.8.0 h1:5SgMzNM5HxrEjV0ww2lTmX6E2Izsfxas4+YHWRs3Lsk= -
github.com/go-swagger/go-swagger
 v0.1.0 h1:6eT+4V5fXHB9w1WNuXFlHC4E3Rj9v6SiXw7knQzRNa0= h1:fOcXeMI1KPNv3uk4u7cR4VSyq0NyrYx4SS1/ajuTWDg=
 v0.2.0 h1:Aa9r3gBPNYHNsmUqBwFPCZ5kPU85gM6MYliecEGihbY= -
 v0.17.0 h1:VXtAcbBmNFxyDYIKp1udAscwbr9Y+/ZP+woNivlsHEY= -
 v0.17.1 h1:vXic4XJ4kMnmt6hfPzHhkfCHd6khpY81oyPLdUEpksQ= -
 v0.17.2 h1:eizwRyO8THHMA4kXyM5Z1UTPslZGE8VsfJC0jJqsRI8= -
 v0.18.0 h1:oVUUoMY2DMfDjjWqBdDaGbdxBfHllH3xt2DSJr7IHOI= -
github.com/go-xorm/core
 v0.5.1 h1:uJNJQxMtdaRSIUcbNz468H0ZtbkNUhZbqnYGHqqU3+0= h1:i7QESCABdFcvhgc8pdINtzlJf/6LC29if6ZJgHt9SHI=
 v0.5.2 h1:+4sNEcNClhkVxyWNIqyzcI+Q7V8c5k62xR3CGYmgJiQ= -
 v0.5.3 h1:LqogwV7BJBq/hj4GHlsv9s7wgTKLPiiqlnhZ1rOn05A= -
 v0.5.4 h1:ebpCY/N1i7a/EkVEYvmQOZreiDNQyTl0BsmplCm0whQ= -
 v0.5.6 h1:SsxP4wBowrZPLhGBDCZtKmVoKhXL1f2NcePPO4NpKss= -
 v0.5.7 h1:ClaJQDjHDre5Yco2MmkWKniM8NNdC/OXmoy2HfxxECw= -
 v0.5.8 h1:vQ0ghlVGnlnFmm4SpHY+xNnPlH810paMcw+Hwz9BCqE= h1:d8FJ9Br8OGyQl12MCclmYBuBqqxsyeedpXciV5Myih8=
 v0.6.0 h1:tp6hX+ku4OD9khFZS8VGBDRY3kfVCtelPfmkgCyHxL0= -
 v0.6.1 h1:ha61NwbKJjPgoLgU+6ajaX1lIQPzeljHkz8c2qrVzxQ= -
 v0.6.2 h1:EJLcSxf336POJr670wKB55Mah9f93xzvGYzNRgnT8/Y= h1:bwPIfLdm/FzWgVUH8WPVlr+uJhscvNGFcaZKXsI3n2c=
github.com/go-xorm/xorm
 v0.5.6 h1:YYSBWOMcnpA7TA7vg8DXBv7zPzMcMBIR67fW3OqM3io= h1:i7qRPD38xj/v75UV+a9pEzr5tfRaH2ndJfwt/fGbQhs=
 v0.5.7 h1:RNgftPrivAioB7+WzY/WDDg8cuvFfqJPdIrGvTxwQXI= -
 v0.5.8 h1:zXgNdrbVaPBjChNxy6qEa0FhIwsyNttEsJ6PhZQXu8A= -
 v0.6.2 h1:G3uSG7i/cjK4FkCLdtyoPxd5cXrUwBl4WhvxA1piOqE= -
 v0.6.3 h1:UxllS2puvBE+0/8o8QvMwRWp6LAXKDo7PP8pA49eonw= -
 v0.6.4 h1:J0X2FjXl6voshSZj91iYPuLsXfvrKJ9NB+Bpq0/kDBc= -
 v0.6.5 h1:42tuMFoEf4/DiDWfRhGzn94TaBtm4SQTfiQdU9D86rw= h1:xkHEPeHG0ckC95t/J9vR4tr4gE4944G/kGnDCGLAxlA=
 v0.6.6 h1:h+5b4/ozenzhEiLMFYeCQO0MGfO9dE+PD1ljPCLIzH4= -
 v0.7.0 h1:u3X44NiN3ggNeeJO3f1xmHIFGAqh3qmCWS3ROuIBWmU= -
 v0.7.1 h1:Kj7mfuqctPdX60zuxP6EoEut0f3E6K66H6hcoxiHUMc= h1:EHS1htMQFptzMaIHKyzqpHGw6C9Rtug75nsq6DA9unI=
github.com/go-yaml/yaml
 v2.0.0+incompatible h1:ColVRwfZI9RI+9BfBZi2plxqB/Mu4ZZzi8RnO15WxXM= h1:w2MrLa16VYP0jy6N7M5kHaCkaLENm+P+Tv+MfurjSw0=
 v2.1.0+incompatible h1:RYi2hDdss1u4YE7GwixGzWwVo47T8UQwnTLB6vQiq+o= -
github.com/goadesign/goa
 v1.0.0 h1:hZrjUA2VbU3NjVEZzLHj7KXW9hCBJXw4wKQHy9KCD+4= h1:d/9lpuZBK7HFi/7O0oXfwvdoIl+nx2bwKqctZe/lQao=
 v1.1.0 h1:mNn6R9jZpAvKADf7diAwrN8H+O6CJtDltxK5o+0Y6ek= -
 v1.2.0 h1:m/B5mwezep9X4c7p0bURPmqD/glmLeP7iZtGUl6FXwg= -
 v1.2.1 h1:nTQaBoILw06uorLsrwhD2DJYz+aGJ35jHTe+tgtvYzo= -
 v1.3.0 h1:7GUDe4L8MBfRw+Rm2F6NWWJm6lRtG8lpxhyNwhoLtaY= -
 v1.3.1 h1:J7gZm0xlJhdQEZ4LWOJUvrwBHM2PA+6ZvMkUq0j0Zms= -
 v1.4.0 h1:2K0k+KtwGcErmWUoarsItuV9cPKG/7KBG86Oh89kHqs= -
 v1.4.1 h1:7klkZZ3eCXewU3E1//C2spxle0dzRRUVdeny/vdKrz4= -
github.com/gobuffalo/packr
 v1.21.1 h1:z+0ev0daZkqrfRrtkT78FntrpxPC+CsSEnT614feV44= h1:1bviu6xjcPv2D1fAyMz1VCESu69O5V3JJXLW9xGDI38=
 v1.21.2 h1:p7n74ZN0vEgueluoe1myi23NoUjztixVQWfkGQ61W6U= h1:IODzKNHqDtRH8Re8hDrLJeOZGJmY+66vmqb1kMgM69k=
 v1.21.3 h1:TJCUm5Kl/Ykl13kZ7KiJHATMtqEiOtQEubywPsAzt0U= -
 v1.21.4 h1:mXkGAx/KDhOerg8z9onddL3E6f90Thevf7QODhpL4ZQ= h1:zCvDxrZzFmq5Xd7Jw4vaGe/OYwzuXnma31D2EbTHMWk=
 v1.21.5 h1:wPOC4Cg5tGYDImw601x94a2d2PEa4KgVHzXMFR2WlBk= -
 v1.21.6 h1:6z9Jk0SD5aAibQn9B+YHkPOuFwvPffhuIk1y5ki3HRE= h1:CxQkdRbcu0aO5mV/Y0hte5jCAFX/4S+cyO5pKCtezG4=
 v1.21.7 h1:2zLx48rSNB4ZmoxyqT1vVOnpo0xoJoGfZWfm/U7xXAs= h1:73tmYjwi4Cvb1eNiAwpmrzZ0gxVA4KBqVSZ2FNeJodM=
 v1.21.8 h1:qvCD8cQzb0MEDTM1xI6l+9R+HNzQ3hNcllUQdCXzcC0= h1:aRZXyERYmMgohDp5wDWnbgn5KiWuCKG19WnWZcAqeII=
 v1.21.9 h1:zBaEhCmJpYy/UdHGAGIC3vO5Uh7RW091le41+Ydcg4E= h1:GC76q6nMzRtR+AEN/VV4w0z2/4q7SOaEmXh3Ooa8sOE=
 v1.22.0 h1:/YVd/GRGsu0QuoCJtlcWSVllobs4q3Xvx3nqxTvPyN0= h1:Qr3Wtxr3+HuQEwWqlLnNW4t1oTvK+7Gc/Rnoi/lDFvA=
github.com/gobwas/glob
 v0.2.1 h1:afOyqqg+bzqIrh0EwMWCGQjQm85wFdVbDlAUAseIPL4= h1:d3Ez4x06l9bZtSvzIay5+Yzi0fmZzPgnTbPcKjJAkT8=
 v0.2.2 h1:czsC5u90AkrSujyGY0l7ST7QVLEPrdoMoXxRx/hXgq0= -
 v0.2.3 h1:A4xDbljILXROh+kObIiy5kIaPYD8e96x1tgBhUI5J+Y= -
github.com/godbus/dbus
 v4.0.0+incompatible h1:iNJ3QcnEtQA2va/vj1d2Ng5Ld6tWno5HscHoVw9Bk/I= h1:/YcGZj5zSblfDWMMoOzV4fas9FZnQYTkDnsGvmh2Grw=
 v4.1.0+incompatible h1:WqqLRTsQic3apZUK9qC5sGNfXthmPXzUZ7nQPrNITa4= -
github.com/gogits/gogs
 v0.11.4 h1:75gzrF+kE02MRahaOukwUB2fMgqiO9zzjp5bbLiveWA= h1:H8FMbPPb+o/TgI6YnmQmT8nmEIHypXDau+f2CChYoCk=
 v0.11.19 h1:y4zFMpYzKBKBJ8DeuCnsAE/EStue8Qb2sP9/IZ455PE= -
 v0.11.29 h1:nDoHeBc4s3ukwgDUthar107l/HMa6+NkhR8+ulqkmts= -
 v0.11.33 h1:M7wkG/hQpAPbD1U6T3q3P/lNEMpmPzPP94Rwr7C2rb0= -
 v0.11.34 h1:X7I71Pr7I2qfRyQACifH6mKAFCQq67417l/FI5cA7Cc= -
 v0.11.43 h1:7buYNxrBzyOsJtTABAIda+rJrFPesS7l+JFmGZi1J+Q= -
 v0.11.53 h1:6MRuZiYgKpVtBLFSA+NkvrT1jPVNTwoYfzpqEASQVHU= -
 v0.11.66 h1:FdgJCjMrSk0qudJIeYKNa4sMZKyyy9Fy8GMSa5frtfo= -
 v0.11.79 h1:dKGlDSSSYtW0R3Y3X/ItenIBrDh48kNB//JPv7+C3mY= -
 v0.11.86 h1:IujCpA+F/mYDXTcqdy593rl2donWakAWoL2HYZn7spw= -
github.com/gogo/protobuf
 v1.0.0 h1:2jyBKDKU/8v3v2xVR2PtiWQviFUyiaGk2rpfyFT8rTM= h1:r8qH/GZQm5c6nD/R0oafs1akxWv10x8SbQlK7atdtwQ=
 v1.1.0 h1:mWrvyIHj8iy7uu+K0BDUbVkUTljYA/az5ziGfhs3ZMA= -
 v1.1.1 h1:72R+M5VuhED/KujmZVcIquuo8mBgX4oVda//DQb3PXo= -
 v1.2.0 h1:xU6/SpYbvkNYiptHJYEDRseDLvYE7wSqhYYNy0QSUzI= -
 v1.2.1 h1:/s5zKNz0uPFCZ5hddgPdo2TK2TVrUNMn0OOX8/aZMTE= h1:hp+jE20tsWTFYpLwKvXlhS1hjn+gTNwPg2I6zVXpSg4=
github.com/gographics/imagick
 v2.2.2+incompatible h1:B8EFzGrE9Am1wgl17UFE8cIvu3Kwxr9w6OAhYXfsWr4= h1:gk55mrttmUSR+Vt6dxnxuD7EKU/3e5Ehe8HRGVCI8mw=
 v2.3.0+incompatible h1:5EM+wqPJjRmxdwRqi5QG9S6+A59mr3GrPvZDd/YvuMg= -
 v2.3.1+incompatible h1:XK+4KGXfwkLMahKcWF6OhsJyS2BNjJapnlamxzJX/PY= -
 v2.4.0+incompatible h1:newMrcrBRmV17WH8szw16BnurCIX2gymiqYu4ulvg4E= -
 v2.4.1+incompatible h1:4VdqQLSuIlMLoib2yI9h62Wz3io7dGoZ9vwC9hUxtto= -
 v2.5.0+incompatible h1:F/ivAAjWwKZ5/xN+AbGbV+ZhJTj6ShDx+Sf3GMp9txQ= -
 v3.0.0+incompatible h1:CUysrDKv8/Qk88GClTz7nQx0Qt9lhjGkEgGz9u1Z4dg= -
 v3.1.0+incompatible h1:YohQp5Wlp1dpRJo7/JsXIk0EEtQBc5VGagtECQaGjFE= -
 v3.1.1+incompatible h1:dGE75aK4kfLLmMBeuWHVqClXYTkVsA+dEYZObCrJ6Q4= -
 v3.2.0+incompatible h1:KeoHcmV7bVRKnj6tXTIcEGnyDg8kO0vPNUTC/5HzwOI= -
github.com/golang/mock
 v1.0.0 h1:HzcpUG60pfl43n9d2qbdi/3l1uKpAmxlfWEPWtV/QxM= h1:oTYuIxOrZwtPieC+H1uAHpcLFnEyAGVDL/k47Jfbm0A=
 v1.0.1 h1:l3aoPzuNMWzLwKntc8zNn3mykKJKCUWVFlxOUGui3/E= -
 v1.1.0 h1:VUon3XjHfsmT+ixZLEDmaSytI4aS3Swyli3SzKxaqKc= -
 v1.1.1 h1:G5FRp8JnTd7RQH5kemVNlMeyXQAztQ3mOWV95KxsXH8= -
 v1.2.0 h1:28o5sBqPkBsMGnC6b4MvE2TzSr5/AT4c/1fLqVGIwlk= -
github.com/golang/protobuf
 v1.0.0 h1:lsek0oXi8iFE9L+EXARyHIjU5rlWIhhTkjDz3vHhWWQ= h1:6lQm79b+lXiMfvg/cZm0SGofjICqVBUtrP5yJMmIC1U=
 v1.1.0 h1:0iH4Ffd/meGoXqF2lSAhZHt8X+cPgkfn/cb6Cce5Vpc= -
 v1.2.0 h1:P3YflyNX/ehuJFLhxviNdFxQPkGK5cDcApsge1SqnvM= -
github.com/golang/snappy
 v0.0.1 h1:Qgr9rKW7uDUkrbSmQeiDsGa8SjGyCOGtuasMWwvp2P4= h1:/XxbfmMg8lxefKM7IXC3fBNl/7bRcc72aCRzEWrmP2Q=
github.com/google/cadvisor
 v0.28.4 h1:5KeHrntd17Bjg7eUUo+8LMSefn1WtUSftEgLxoUAkXE= h1:1nql6U13uTHaLYB8rLS5x9IJc2qT6Xd/Tr1sTX6NE48=
 v0.28.5 h1:aNE2bzEZ/gl0z0OYAbqJu2HhZsozab0uhsGWlj806bI= -
 v0.29.0 h1:V5QsHLEF1cgF+gIj4hvKs5dsbxEZbq/6Hi21vD3jRlo= -
 v0.29.1 h1:uHkr9/ieyRszJ9IYt9zL3M82Bimu/Gn+Z6mexQFkyFQ= -
 v0.29.2 h1:RgZwByLZ7eUEcNpaDojYRvYnkC2EbDpc49vRwvcfBDQ= -
 v0.30.0 h1:kiZsBEUoj11nQul9QiYuScMdS++c//Rgn/KczqowKsE= -
 v0.30.1 h1:e5wsh7JIwXVQ9vhf76h2gtpHk6yp3KID2Z9udXYgY6Q= -
 v0.30.2 h1:f9JTPxuV3OKKG8jPhhH1h6j2xIh/tqAj88xeG69Dr+U= -
 v0.31.0 h1:7IEpDDIrh2bpOVyICb2iItyNVnMKaXN37HSDOTqzv/k= -
 v0.32.0 h1:eMoAOoZmuRMZ/yryNW1Fyu4wUsUWyp3UQ59s+A+qrPI= -
github.com/google/flatbuffers
 v1.2.0 h1:uVbD0DOKYVqou9kovj8SnSsqt1HXM1Ez2DQRrlJljII= h1:1AeVuKshWv4vARoZatz6mlQ0JxURH0Kv5+zNeJKJCa8=
 v1.3.0 h1:TKXcb2mXp+B3azt0ZYuWPmLBFgIDGAH7WEhZn50m4q4= -
 v1.4.0 h1:QkMronN85wKgNpCxjbh3jE+24g1llgcJdMqsLw3Iq+k= -
 v1.5.0 h1:rjr1PsU/1It22AJQxqGD4MEAgbT1je80Vy2L5YMhOmw= -
 v1.6.0 h1:gtMr1neQZSJ7c5cN+hYjAbD+GaDUd/7Gdug1rZdr+uM= -
 v1.7.0 h1:7+nRbfwRQZKgfSjUyqBzYsv43tHo6NMRIw78i1q+pMI= -
 v1.7.1 h1:A+4GKBJCIyGRlrxA2Pl7qG7qr7JiBZghSsK+rzw6QNQ= -
 v1.8.0 h1:CdnRsHiH1T8RQa9ytSGmiVRyEXr9LbqCkMx210bAueM= -
 v1.9.0 h1:ZncsT6mBICwWsuCMmJ0OOasV1qpUWIGBWX5e66qIW38= -
 v1.10.0 h1:wHCM5N1xsJ3VwePcIpVqnmjAqRXlR44gv4hpGi+/LIw= -
github.com/google/go-github
 v8.0.0+incompatible h1:f9cxKuLChkifpGhFZhTWXcUXssIVeK/b+9BsS4RGe5U= h1:zLgOLi98H3fifZn+44m+umXrS52loVEgC2AApnigrVQ=
 v9.0.0+incompatible h1:US9GklxPFtBOBUz+pdGJ5Gy6zKkUc+/bDc0ZD/7cSDk= -
 v10.0.0+incompatible h1:3+5U8RmC69FMl2NDNBQVITYxfPgCC8adS/9UcEXrOhY= -
 v11.0.0+incompatible h1:2E2ox5VD0L/xIOFzpo7+fJzWFL5H0J8gmbBsDF23vWs= -
 v12.0.0+incompatible h1:CORPvYkD1fpIoiZ7wkRk+m+WhN7Yhx+6CWG2zFYPn2k= -
 v13.0.0+incompatible h1:fBlwo+i9Kp/7PQvRin1IVtHidPs+tLT3rVVFhrrK1+o= -
 v14.0.0+incompatible h1:IH7XxuaXbLVh4iwPks5+jmKZXElyvAf+5K1108Ku8fU= -
 v15.0.0+incompatible h1:jlPg2Cpsxb/FyEV/MFiIE9tW/2RAevQNZDPeHbf5a94= -
 v16.0.0+incompatible h1:omSHCJqM3CNG6RFFfGmIqGVbdQS2U3QVQSqACgwV1PY= -
 v17.0.0+incompatible h1:N0LgJ1j65A7kfXrZnUDaYCs/Sf4rEjNlfyDHW9dolSY= -
github.com/google/go-querystring
 v1.0.0 h1:Xkwi/a1rcvNg1PPYe5vI8GbeBY/jrVuDX5ASuANWTrk= h1:odCYkC5MyYFN7vkCjXpyrEuKhc/BUO6wN/zVPAxq5ck=
github.com/google/gopacket
 v1.1.7 h1:pVZ9hBdPwoTjkupZ1S2XZB2qwDqbhO/AAw/BvF9u2qE= h1:UCLx9mCmAwsVbn6qQl1WIEt2SO7Nd2fD0th1TBAsqBw=
 v1.1.8 h1:lzrpxBfEQwmhufpcDwAaXZrpocxrYPt47xv/iEpMBCo= -
 v1.1.9 h1:ML//DCFcH2W9Oe1ALVJEXFYu2KDjYJs3+h6+L3RPRws= -
 v1.1.10 h1:FgixKO2r2AJYhYFKNgDGV4Kv7Gr9+BP24oUOPoOR4K0= -
 v1.1.11 h1:aqsawFLqw0XKVA6Zbmk5dhcpS7Qz/aIrsH4Og1osvNA= -
 v1.1.12 h1:iO4CXvFUmDaulbkZcejmXgrZNobyLkgBtYwgvA2hD6Y= -
 v1.1.13 h1:MftQ/BkkVd5AlHcMnraUVErj9zZe2IN2IuHq6q+7TCU= -
 v1.1.14 h1:1+TEhSu8Mh154ZBVjyd1Nt2Bb7cnyOeE3GQyb1WGLqI= -
 v1.1.15 h1:M6W3hwQXo5rq1wyhRByGhqOw0m9p+HWtUJ3Bj4/fT6E= -
 v1.1.16 h1:u6Afvia5C5srlLcbTwpHaFW918asLYPxieziOaWwz8M= -
github.com/google/gops
 v0.0.1 h1:Zu4mP3q+A3CTzFeEnZdB/O8G2N+qYAG9X/m7MDs7eyw= h1:pMQgrscwEK/aUSW1IFSaBPbJX82FPHWaSoJw1axQfD0=
 v0.1.0 h1:eq78BpQ7Tn+T8ppwm/ww7zzpq8aR/Cpghdo/mntT3x8= -
 v0.2.0 h1:Ehgp7wk/6UkgtRbpHue/YDgGwbISiwJFbk83J+8Y64k= -
 v0.3.0 h1:6iywf8lJB4yD8w1s4iOPm/GND76QDUex3Y9CkOh0YaI= -
 v0.3.1 h1:kfHLNJKQ3awjI8K8Zwkqf0KdVLxiJKAAVR6sAibslRo= -
 v0.3.2 h1:n9jMkrye8dh3WQ0IxG5dzLRIhQeZDZoGaj0D7T7x7hQ= -
 v0.3.3 h1:QTgQ3WE0hSRQmU6aAyeePI+l9BI60qvr6xp4J2oKsGs= -
 v0.3.4 h1:RpHu+onj/uS84Xry+4n8W6UMwkLBOvysUAlDsF3rflo= -
 v0.3.5 h1:SIWvPLiYvy5vMwjxB3rVFTE4QBhUFj2KKWr3Xm7CKhw= -
 v0.3.6 h1:6akvbMlpZrEYOuoebn2kR+ZJekbZqJ28fJXTs84+8to= h1:RZ1rH95wsAGX4vMWKmqBOIWynmWisBf4QFdgT/k/xOI=
github.com/google/subcommands
 v1.0.1 h1:/eqq+otEXm5vhfBrbREPCSVQbvofip6kIz+mX5TUH7k= h1:ZjhPrFU+Olkh9WazFPsl27BQ4UPiG37m3yTrtFlrHVk=
github.com/google/uuid
 v1.0.0 h1:b4Gk+7WdP/d3HZH8EJsZpvV7EtDOgaZLtnaNGIu1adA= h1:TIyPZe4MgqvfeYDBFedMoGGpEw/LqOeaOT+nhxU+yHo=
 v1.1.0 h1:Jf4mxPC/ziBnoPIdpQdPJ9OeiomAUHLvxmPRSPH9m4s= -
github.com/googollee/go-engine.io
 v1.0.1 h1:Q0H6NyghLSleyzQa5pN7N0ZZw15MLcgd+kqgXM2eAcA= h1:ZcJSV0EqRvvcCXN7h7d8/EncnShfx85kv0SUsTIKTsg=
 v1.4.1 h1:m3WlZAug1SODuWT++UX2nbzk9IUCn9T1SnmHoqppdqo= h1:26oFqHsnuWIzNOM0T08x21eQOydBosKOCgK3tyhzPPI=
github.com/googollee/go-socket.io
 v0.9.1 h1:KYsu63c3H5SaeQ3MDlHSTE/LJnwok2SH1M5wy4ZaYD0= h1:Q0CvnKmaZNgDXIi85at4eLadAOS1hWDLaDATQpuH3i4=
 v1.0.1 h1:uWBxm1BBV7XSFHOr0vZMYC6TMHMPsI3YcQPXMWzwjUw= h1:I46rLznx5OmtL5sPHp9GQJK/z0+lkLOBIx1NO8Mp5io=
 v1.4.1 h1:lnMJhKTvXKsmSssVjJPzuglU5y6Bf9SizA7s9XZvyqw= h1:yjlQxKcAZXZjpGwQVW/y1sgyL1ou+DdCpkswURDCRrU=
github.com/gorilla/context
 v1.1.1 h1:AWwleXJkX/nhcU9bZSnZoi3h/qGYqQAGhq6zZe/aQW8= h1:kBGZzfjB9CEq2AlWe17Uuf7NDRt0dE0s8S51q0aT7Yg=
github.com/gorilla/csrf
 v1.0.1 h1:QxBz2bWSKSPdjFolloK8iS/bXkq30/mMR6KTEdRUvvk= h1:hxGa+qNn35co03vt75oDkIVPid4opvgJdE8E7yK0qKs=
 v1.0.2 h1:LPPcehmb1zu82xln/wQN8/bVZwlEojv3J2ENGKgf8N0= -
 v1.5.1 h1:UASc2+EB0T51tvl6/2ls2ciA8/qC7KdTO7DsOEKbttQ= h1:HTDW7xFOO1aHddQUmghe9/2zTvg7AYCnRCs7MxTGu/0=
github.com/gorilla/feeds
 v1.0.0 h1:EbkEvaYf+PXhYNHS20heBG7Rl2X6Zy8l11ZBWAHkWqE= h1:Nk0jZrvPFZX1OBe5NPiddPw7CfwF6Q9eqzaBbaightA=
 v1.1.0 h1:pcgLJhbdYgaUESnj3AmXPcB7cS3vy63+jC/TI14AGXk= -
github.com/gorilla/handlers
 v1.2.1 h1:IW0s9JrxTVsutEp77dGDlBv+PZnW6HKse4TrzJ0b+8g= h1:Qkdc/uu4tH4g6mTK6auzZ766c4CA0Ng8+o/OAirnOIQ=
 v1.3.0 h1:tsg9qP3mjt1h4Roxp+M1paRjrVBfPSOpBuVclh6YluI= -
 v1.4.0 h1:XulKRWSQK5uChr4pEgSE4Tc/OcmnU9GJuSwdog/tZsA= -
github.com/gorilla/mux
 v1.2.0 h1:3XN1wbFJAJzEqzeUqlVF0qgpqZFHfV1YNHYPU+odUnw= h1:1lud6UwP+6orDFRuTfBEV8e9/aOM/c4fVVCaMa2zaAs=
 v1.3.0 h1:HwSEKGN6U5T2aAQTfu5pW8fiwjSp3IgwdRbkICydk/c= -
 v1.4.0 h1:N6R8isjoRv7IcVVlf0cTBbo0UDc9V6ZXWEm0HQoQmLo= -
 v1.5.0 h1:mq8bRov+5x+pZNR/uAHyUEgovR9gLgYFwDQIeuYi9TM= -
 v1.6.0 h1:UykbtMB/w5No2LmE16gINgLj+r/vbziTgaoERQv6U+0= -
 v1.6.1 h1:KOwqsTYZdeuMacU7CxjMNYEKeBvLbxW+psodrbcEa3A= -
 v1.6.2 h1:Pgr17XVTNXAk3q/r4CpKzC5xBM/qW1uVLV+IhRZpIIk= -
 v1.7.0 h1:tOSd0UKHQd6urX6ApfOn4XdBMY6Sh1MfxV3kmaazO+U= -
github.com/gorilla/rpc
 v1.1.0 h1:marKfvVP0Gpd/jHlVBKCQ8RAoUPdX7K1Nuh6l1BNh7A= h1:V4h9r+4sF5HnzqbwIez0fKSpANP0zlYd3qR7p36jkTQ=
github.com/gorilla/schema
 v1.0.0 h1:RhePNm5bGqvwu58UVVa/sjPOc5C60aOhZgBHWQTh4p4= h1:hJqhTosYf3R5XsKxm+0dOSxCiNiJOQB/7ajkOstcQRQ=
 v1.0.1 h1:SRdkNFH8S7c3v2UCiVUUNhnBsJmWwiKboYbyODLYPX8= h1:kgLaKoK1FELgZqMAVxx/5cbj0kT+57qxUrAlIO2eleU=
 v1.0.2 h1:sAgNfOcNYvdDSrzGHVy9nzCQahG+qmsg+nE8dK85QRA= -
github.com/gorilla/securecookie
 v1.1.1 h1:miw7JPhV+b/lAHSXz4qd/nN9jRiAFV5FwjeKyCS8BvQ= h1:ra0sb63/xPlUeL+yeDciTfxMRAA+MP+HVt/4epWDjd4=
github.com/gorilla/sessions
 v1.1.1 h1:YMDmfaK68mUixINzY/XjscuJ47uXFWSSHzFbBQM0PrE= h1:8KCfur6+4Mqcc6S0FEfKuN15Vl5MgXW92AE8ovaJD0w=
 v1.1.2 h1:4esMHhwKLQ9Odtku/p+onvH+eRJFWjV4y3iTDVWrZNU= -
 v1.1.3 h1:uXoZdcdA5XdXF3QzuSlheVRUvjl+1rKY7zBXL68L9RU= -
github.com/gorilla/websocket
 v1.0.0 h1:J/mA+d2LqcDKjAEhQjXDHt9/e7Cnm+oBUwgHp5C6XDg= h1:E7qHFY5m1UJ88s3WnNqhKjPHQ0heANvMoAMk2YaljkQ=
 v1.1.0 h1:IhvMPOB8GxycsyOkvML1FrwAFiKgfHlS9KWKa6EqE6Q= -
 v1.2.0 h1:VJtLvh6VQym50czpZzx07z/kw9EgAxI3x1ZB8taTMQQ= -
 v1.3.0 h1:r/LXc0VJIMd0rCMsc6DxgczaQtoCwCLatnfXmSYcXx8= -
 v1.4.0 h1:WDFjx/TMzVgy9VdMMQi2K2Emtwi2QcUQsztZ/zLaH/Q= -
github.com/gosimple/slug
 v1.0.1 h1:dThmvIXq3immCGopPdm7WfY7p4eV8snrFOknZTEH0qw= h1:ER78kgg1Mv0NQGlXiDe57DpCyfbNywXXZ9mIorhxAf0=
 v1.0.2 h1:cgDNdhrOlWqSdXN0wdyVJHRFSy73A6HfCcoI6tuGbW0= -
 v1.0.3 h1:xW8VbiBa1X4dOJYeXDjhAJWcjvm+lxn3QLT8OGbuJfo= -
 v1.1.0 h1:qazuiO8Pq90VbYCUh8/iNY8ShifnkXhRY6LOT8ZtNa4= -
 v1.1.1 h1:fRu/digW+NMwBIP+RmviTK97Ho/bEj/C9swrCspN3D4= -
 v1.2.0 h1:DqQXHQLprYBsiO4ZtdadqBeKh7CFnl5qoVNkKkVI7No= -
 v1.3.0 h1:NKQyQMjKkgCpD/Vd+wKtFc7N60bJNCLDubKU/UDKMFI= -
 v1.4.0 h1:CorzyNkphIu/RJawagGblB7M+aXakjt/MhuXvlKEb98= -
 v1.4.1 h1:h29PRcKc8dPN//lJ9Ib6EKP50kG5AmpJ0yRjn7ksY/8= -
 v1.4.2 h1:jDmprx3q/9Lfk4FkGZtvzDQ9Cj9eAmsjzeQGp24PeiQ= -
github.com/grafana/grafana
 v5.3.3+incompatible h1:rpxeRqGiz/D4sXCQS4iqiUzpbh6KOClIzNKpjGY90U8= h1:U8QyUclJHj254BFcuw45p6sg7eeGYX44qn1ShYo5rGE=
 v5.3.4+incompatible h1:Ue+crTJrQOgINdZ0yGt42UAjynBe2ReUf269R8W/RRA= -
 v5.4.0-beta1+incompatible h1:VceOjZ2qjLd+YttyhQPItoqCXyEr9kL3zHtQn0wTVoA= -
 v5.4.0+incompatible h1:1xxt2e94UZGGBB9pvt9R84VwY1pQbkk5r6rKLyOdo/U= -
 v5.4.1+incompatible h1:eXUNXv61BVoydxcIfuRRbEAAMunTYoTpy0a3EtyFWdw= -
 v5.4.2+incompatible h1:IBmlh0FKISC6Pj6F6zwJsvP+XkBWZevBDqprhGLwYnY= -
 v5.4.3+incompatible h1:+nT+ADXgOsP9YC3uT1ydiFC7fTKkDxOPpPEcqVlGoyI= -
 v6.0.0-beta1+incompatible h1:8VPhbr1F231OGses3HS8mZqsoqeHmAzQ1/0AVaTfBcI= -
 v6.0.0-beta2+incompatible h1:EiZRU+yoarswnYONeXl/FNCyGg2HZqBYPqMLR1fKP1Q= -
 v6.0.0-beta3+incompatible h1:o1pF4HDWk0kpO6s70m2fIVG3rVtWotphPabrUW2A5eQ= -
github.com/graphql-go/graphql
 v0.5.0 h1:ssPB6Byi+mfBm6zI6yRQhbgHQQJJMh7qgBf+B1aBu/o= h1:k6yrAYQaSP59DC5UVxbgxESlmVyojThKdORUqGDGmrI=
 v0.6.0 h1:GfvaRDnzkyp160/2WebwvLiXCaAJci87G/FSsFwe4zY= -
 v0.7.0 h1:d8FJNktjN7z252InPq3Esq30EjP3Wo1DFxiGSyMS90k= -
 v0.7.1 h1:hRcwvLRSINzgwLp8SqHtzUlAjUvIkN6SqJJvIrPMQKk= -
 v0.7.2 h1:taAtizI+aQQE8b5DVhylo/KvBVm2KfAgfjxv48loamA= -
 v0.7.3 h1:+nNk2hSYiJeEEzZFwlEuX78iPyRrLXuw2tANQo0ATn8= -
 v0.7.4 h1:mXfK88XYicw3O0bfPAgiWb50iHGgXatlcxVbr2p3AY8= -
 v0.7.5 h1:/JYC+NCUsSAfP/bVn1/ij8zvc7kzLwXUMyctSXdsE6o= -
 v0.7.6 h1:3Bn1IFB5OvPoANEfu03azF8aMyks0G/H6G1XeTfYbM4= -
 v0.7.7 h1:nwEsJGwPq9N6cElOO+NYyoWuELAQZ4GuJks0Rlco5og= -
github.com/grpc-ecosystem/go-grpc-middleware
 v1.0.0 h1:Iju5GlWwrvL6UBg4zJJt3btmonfrMlCDdsejg4CZE7c= h1:FiyG127CGDf3tlThmgyCl78X/SZQqEOJBCDaAfeWzPs=
github.com/grpc-ecosystem/go-grpc-prometheus
 v1.2.0 h1:Ovs26xHkKqVztRpIrF/92BcuyuQ/YW4NSIpoGtfXNho= h1:8NvIoxWQoOIhqOTXgfV/d3M/q6VIi02HzZEHgUlZvzk=
github.com/grpc-ecosystem/grpc-gateway
 v1.4.0 h1:imhhuBJyLcvIi1OHmWDyrOMJP/A7mgmhf7GszdyhDpY= h1:RSKVYQBd5MCa4OVpNdGskqpgL2+G+NZTnrVHpWWfpdw=
 v1.4.1 h1:pX7cnDwSSmG0dR9yNjCQSSpmsJOqFdT7SzVp5Yl9uVw= -
 v1.5.0 h1:WcmKMm43DR7RdtlkEXQJyo5ws8iTp98CyhCCbOHMvNI= -
 v1.5.1 h1:3scN4iuXkNOyP98jF55Lv8a9j1o/IwvnDIZ0LHJK1nk= -
 v1.6.0 h1:MQ2oj/ms4WsaAC1GT9BG1DFsxcyV5N1b9FdWunXAT0o= -
 v1.6.1 h1:N6Z6yCkj/XfYGhTRfxEhInVcslxlfw4Bw+Di3GqW5aM= -
 v1.6.2 h1:8KyC64BiO8ndiGHY5DlFWWdangUPC9QHPakFRre/Ud0= -
 v1.6.3 h1:oQ+8y59SMDn8Ita1Sh4f94XCUVp8AB84sppXP8Qgiow= -
 v1.6.4 h1:xlu6C2WU6gvXt3XLyVpsgweaIL4VCmTjEsEAIt7qFqQ= -
 v1.7.0 h1:tPFY/SM+d656aSgLWO2Eckc3ExwpwwybwdN5Ph20h1A= -
github.com/guregu/null
 v2.0.1+incompatible h1:DQCDBQ7g9VnyH1cuOHIoEUDCIWdJ/FoHAd2GsqM0ppk= h1:ePGpQaN9cw0tj45IR5E5ehMvsFlLlQZAkkOXZurJ3NM=
 v2.1.1+incompatible h1:R9lg8cS85qvWsfz3RW1P0M8AwXRUae+FywE974nNzuU= -
 v2.1.2+incompatible h1:6wQTdgkdM4H49tC7s1OA9qQJAEc1WH4idfoPJJEHCNQ= -
 v3.0.1+incompatible h1:pGw2Be81Bkw9RjMSLSuTZaTLaODKbCl9hOmT7ioSlLI= -
 v3.2.0+incompatible h1:wNkiGfCmoZzsCAX/NQ+1eWe7CdkpEJA8fQUiBPjGZqU= -
 v3.2.1+incompatible h1:Nfu8DCXnfcMAPGaLwWn/qgoBeAuqeBEEybfG2OkAemU= -
 v3.3.0+incompatible h1:egMb2dwXrFlTRgp6z4LBgmXOEvaUVqYWTbL2J4EX4g0= -
 v3.4.0+incompatible h1:a4mw37gBO7ypcBlTJeZGuMpSxxFTV9qFfFKgWxQSGaM= -
github.com/ha/doozer
 v0.3.1 h1:n37/6LjN+T5T8JAaNb5gZ/dNBEP+GGody0qG/cTJNwk= h1:6eul7yLUgIgrt0BdZFIqPh67Y8VWa3m0gQOgOORTsDE=
github.com/ha/doozerd
 v0.3.1 h1:PYEbLNXslJmWJYltOxFoZXp5SK1ld7EWQfjjC+6pV1Q= h1:8U6Qw56qVXVVZAh3HRPVDWST0TILRtlBys3CvXM0wCQ=
github.com/hashicorp/consul
 v1.2.1 h1:66MuuTfV4aOXTQM7cjAIKUWFOITSk4XZlMhE09ymVbg= h1:mFrjN1mfidgJfYP1xrJCF+AfRhr6Eaqhb2+sfyn/OOI=
 v1.2.2 h1:C5FurAZWLQ+XAjmL9g6rXbPlwxyyz8DvTL0WCAxTLAo= -
 v1.2.3 h1:ekX+fXQ7NYzD2quCCgmDekCCIp0Fsi1NE0ViC2CJm+8= -
 v1.2.4 h1:QgwJnJBs9zuhZN8cCsxrFgT+8HS7TSv49XlwFI5UGVU= -
 v1.3.0 h1:0ihJs1J8ejURfAbwhwv+USnf4oyqfAddv/3xXXv4ltg= -
 v1.3.1 h1:bY7/Uo29Uq7+mHce4wgSHtAJSbeRl+4F7M+OHTuEeXI= -
 v1.4.0-rc1 h1:vEYtR3Y6ENrcl3nMeb38JU0Kj8gnnIs9vhVufAKlUAQ= -
 v1.4.0 h1:PQTW4xCuAExEiSbhrsFsikzbW5gVBoi74BjUvYFyKHw= -
 v1.4.1 h1:yC/A2RW0kWJIlr/VUPwHI7UASngT178VTTIo15S4Wj4= -
 v1.4.2 h1:D9iJoJb8Ehe/Zmr+UEE3U3FjOLZ4LUxqFMl4O43BM1U= -
github.com/hashicorp/errwrap
 v1.0.0 h1:hLrqtEDnRye3+sgx6z4qVLNuviH3MR5aQ0ykNJa/UYA= h1:YH+1FKiLXxHSkmPseP+kNlulaMuP3n2brvKWEqk/Jc4=
github.com/hashicorp/go-checkpoint
 v0.5.0 h1:MFYpPZCnQqQTE18jFwSII6eUQrD/oxMFp3mlgcqk5mU= h1:7nfLNL10NsxqO4iWuW6tWW0HjZuDrwkBuEQsVcpCOgg=
github.com/hashicorp/go-cleanhttp
 v0.5.0 h1:wvCrVc9TjDls6+YGAF2hAifE1E5U1+b4tH6KdvN3Gig= h1:JpRdi6/HCYpAwUzNwuwqhbovhLtngrth3wmdIIUrZ80=
github.com/hashicorp/go-getter
 v1.0.0 h1:J4JXDg6kELIskI+qk1MRrlV7qBpa3UDmvo58Z3VcdEM= h1:eLvWiwRFYGj6M4qeM/PP6Fd0ANDRRxv+xrAiBROanoQ=
 v1.0.1 h1:WlFPjyPrd34KmTQMzSPA0pn9JpRsuHjHaRTx0VPzxYw= h1:tkKN/c6I/LRSXLOWZ8wa/VB0LfVrryHzk/B0aZLKZI0=
 v1.0.2 h1:ba+UwCRuxJ7+rS+cO6JnQZUrweQjmEAkwKu9r7+HCpM= h1:q+PoBhh16brIKwJS9kt18jEtXHTg2EGkmrA9P7HVS+U=
 v1.0.3 h1:CelOrh4nPI/kzBsweEXM8f1dZFNSf1jH4ReJZXWHymY= -
 v1.1.0 h1:iGVeg7L4V5FTFV3D6w+1NAyvth7BIWWSzD60pWloe2Q= -
github.com/hashicorp/go-immutable-radix
 v1.0.0 h1:AKDB1HM5PWEA7i4nhcpwOrO2byshxBjXVn/J/3+z5/0= h1:0y9vanUI8NX6FsYoO3zeMjhV/C5i9g4Q3DwcSNZ4P60=
github.com/hashicorp/go-msgpack
 v0.5.0 h1:rKqhU6VO42cMi9LhhAreNOAUzQa5zdqFl+TUjG7kkUo= h1:ahLV/dePpqEmjfWmKiqvPkv/twdG7iPBM1vqhUKIvfM=
 v0.5.1 h1:hwMd9IlnlQ6jGCBjyhgHZwPy3u95IIGFjejq79Lltus= -
 v0.5.2 h1:VPpzMUjr5KSqptUv4i3bt7VCZH2xOyc3TUiEtkgL7oc= -
 v0.5.3 h1:zKjpN5BK/P5lMYrLmBHdBULWbJ0XpYR+7NGzqkZzoD4= -
github.com/hashicorp/go-multierror
 v1.0.0 h1:iVjPR7a6H0tWELX5NxNe7bYopibicUzc7uPribsnS6o= h1:dHtQlpGsu+cZNNAkkCN/P3hoUDHhCYQXV3UM06sGGrk=
github.com/hashicorp/go-retryablehttp
 v0.5.0 h1:aVN0FYnPwAgZI/hVzqwfMiM86ttcHTlQKbBVeVmXPIs= h1:9B5zBasrRhHXnJnui7y6sL7es7NDiJgTc6Er0maI1Xs=
 v0.5.1 h1:Vsx5XKPqPs3M6sM4U4GWyUqFS8aBiL9U5gkgvpkg4SE= -
 v0.5.2 h1:AoISa4P4IsW0/m4T6St8Yw38gTl5GtBAgfkhYh1xAz4= -
github.com/hashicorp/go-rootcerts
 v1.0.0 h1:Rqb66Oo1X/eSV1x66xbDccZjhJigjg0+e82kpwzSwCI= h1:K6zTfqpRlCUIjkwsN4Z+hiSfzSTQa6eBIzfwKfwNnHU=
github.com/hashicorp/go-syslog
 v1.0.0 h1:KaodqZuhUoZereWVIYmpUgZysurB1kBLX2j0MwMrUAE= h1:qPfqrKkXGihmCqbJM2mZgkZGvKG1dFdvsLplgctolz4=
github.com/hashicorp/go-uuid
 v1.0.0 h1:RS8zrF7PhGwyNPOtxSClXXj9HA8feRnJzgnI1RJCSnM= h1:6SBZvOh/SIDV7/2o3Jml5SYk/TvGqwFJ/bN7x4byOro=
 v1.0.1 h1:fv1ep09latC32wFoVwnqcnKJGnMSdBanPczbHAYm1BE= -
github.com/hashicorp/go-version
 v1.0.0 h1:21MVWPKDphxa7ineQQTrCU5brh7OuVVAzGOCnnCPtE8= h1:fltr4n8CU8Ke44wwGCBoEymUuxUHl09ZGVZPK5anwXA=
 v1.1.0 h1:bPIoEKD27tNdebFGGxxYwcL4nepeY4j1QP23PFRGzg0= -
github.com/hashicorp/golang-lru
 v0.5.0 h1:CL2msUPvZTLb5O648aiLNJw3hnBxN2+1Jq8rCOH9wdo= h1:/m3WP610KZHVQ1SGc6re/UDhFvYD7pJ4Ao+sR/qLZy8=
github.com/hashicorp/hcl
 v1.0.0 h1:0Anlzjpi4vEasTeNFn2mLJgTSwt0+6sfsiTG8qcWGx4= h1:E5yfLk+7swimpb2L/Alb/PJmXilQ/rhwaUYs4T20WEQ=
github.com/hashicorp/logutils
 v1.0.0 h1:dLEQVugN8vlakKOUE3ihGLTZJRB4j+M2cdTm/ORI65Y= h1:QIAnNjmIWmVIIkWDTG1z5v++HQmx9WQRO+LraFDTW64=
github.com/hashicorp/memberlist
 v0.1.0 h1:qSsCiC0WYD39lbSitKNt40e30uorm2Ss/d4JGU1hzH8= h1:ncdBp14cuox2iFOq3kDiquKU6fqsTBc3W6JvZwjxxsE=
 v0.1.1 h1:JCBIrGIyaWQDNTHJlCNCUCOvJ08T4JPKc0Uc9xcovvM= h1:IsMNiIwroSErOGvihcN8kglKKIvAJNzJ5P7H9/XVPH0=
 v0.1.2 h1:q7yR+4E1wvgjjAveOPdwxgEQd60Z5jCCF8pz6Zb6rJ8= h1:n8uF7k+gpteS2BrOQVk2qkAKdCDc7YdEaD6nLpNFBVo=
 v0.1.3 h1:EmmoJme1matNzb+hMpDuR/0sbJSUisxyqBGG676r31M= h1:ajVTdAv/9Im8oMAAj5G31PhhMCZJV2pPBoIllUwCN7I=
github.com/hashicorp/nomad
 v0.8.2 h1:xNAxQopnKjh1N/9IxS961n1emXl8XYz3LLxrR5TGZ1c= h1:WRaKjdO1G2iqi86TvTjIYtKTyxg4pl7NLr9InxtWaI0=
 v0.8.3 h1:5iQUQh3TtTrpSdQwYOgkYjI+Uvky8cjAzRlKIbbUvOQ= -
 v0.8.4-rc1 h1:15NnM82R4304YeJs9QQtxY9eVOAKJ8Ih+zGu+iIg7b0= -
 v0.8.4 h1:KKJjP24Q0hR+JFsFEmegPolmBdRuYy1OfSyfkuWJm7w= -
 v0.8.5 h1:TKPulZ7YCnTOGKMZUgoQ0f0aeV0ZT95gygr1hAGA+jw= -
 v0.8.6 h1:z+gocir324zUa88k9bIXkf0RpSgjVa9Izut+iV8T2qg= -
 v0.8.7-rc1 h1:vI/JMQMxVn+55fcKJFqH8RJlxvreKuAzl2wdj780M08= -
 v0.8.7 h1:jOrmJdAoWcyhKgoG4OxHQhG5SU6RniXFjfwKg6a492U= -
 v0.9.0-beta1 h1:3u4LxFJNYSd4QR5sPvMRzmYEjfSRKI5VF2wHLcSUDMw= -
 v0.9.0-beta2 h1:haKHe9AyERzoRmiliggz/MBhy9STxlGRIQAhgnFVJEA= -
github.com/hashicorp/raft
 v0.1.0 h1:OC+j7LWkv7x8s9c5wnXCEgtP1J0LDw2fKNxUiYCZFNo= h1:DVSAWItjLjTOkVbSpWQ0j0kUADIvDaCtBxIcbNAQLkI=
 v1.0.0 h1:htBVktAOtGs4Le5Z7K8SF5H2+oWsQFYVmOgH5loro7Y= -
github.com/hashicorp/serf
 v0.5.0 h1:pm/nZuf8B94zAdyG7jFurIBBBzJFj/pCl+A/KYBB3mY= h1:h/Ru6tmZazX7WO/GDmwdpS975F019L4t5ng5IgwbNrE=
 v0.6.0 h1:wZ0XaY2sDuq3henygf39+1Q+QK9A/4E69rfW6mceYX4= -
 v0.6.1 h1:1bDcTHYMxgFP73W40g3TRb/hx9zEgTMpfs+6ZhzafUc= -
 v0.6.2 h1:U8rjNQjMNrq/Zo+MbOVENu0diY3ilumB6kS2zyPCjZY= -
 v0.6.3 h1:e8/w+e1zWM48MmJQDtktZNYCi5DA8vFVtPJWpPrEOoY= -
 v0.6.4 h1:j5zxUB+BLvSuEkKOg70qstcmlusStXSPQ2gbm/+2F0Q= -
 v0.7.0 h1:9NBx0ZSoEtMrWrNf2ByXsmoKcpJyHIU7xGU/itMCtkg= -
 v0.8.0 h1:mRqNot7hGnOdAkmxvrN9U0tpdQ5Shlb5uPt4hdJou2U= -
 v0.8.1 h1:mYs6SMzu72+90OcPa5wr3nfznA4Dw9UyR791ZFNOIf4= -
 v0.8.2 h1:YZ7UKsJv+hKjqGVUUbtE3HNj79Eln2oQ75tniF6iPt0= h1:6hOLApaqBFA1NXqRQAsxw9QxuDEvNxSQRwA/JwenrHc=
github.com/hashicorp/terraform
 v0.11.8 h1:fxt+ihK6GZA8Qj2OSNW/j5hn8sE7pFMFS6/HTseTBJI= h1:uN1KUiT7Wdg61fPwsGXQwK3c8PmpIVZrt5Vcb1VrSoM=
 v0.11.9-beta1 h1:LIGWX0BoPCruLMLddpG1u8cKWx0DdvsHHWN/oOjZf8w= -
 v0.11.9 h1:dL0cB5xsIg/rjySx1TVEnVg/FWdB03+jcKJ0O48FqPI= -
 v0.11.10 h1:XYZ/V+teSckRfZdwxsoP+66v0slHixe5Ai8wxqY/lfo= -
 v0.11.11 h1:5q1y/a0RB1QmKc1n6E9tnWQqPMb+nEb7Bfol74N2grw= -
 v0.11.12-beta1 h1:fJPJO6oahPiYKYXnTBYQsnCJxCILwv8ok6up5nL2+s8= -
 v0.12.0-alpha1 h1:jEDfX7EB2yx2wpxZS9n5O+hYbLQsBmQJkoIQT8/cQPw= h1:YvER8PBvJQTIG75o/uecN8Rs11kIqhwgJotgIMnVZJ4=
 v0.12.0-alpha2 h1:shMRFOzg0jaFskaG6emxbR1M7AjQlFYDPrfxZvc9tcA= h1:c16IVVdhvB0IRH1YiqIQMLnyip/wAgzirH6UMkWJruE=
 v0.12.0-alpha3 h1:lNsnXZ2luEf4DnmLw7pXoXGzazXkp1US97+PZYn1x3o= h1:9NILwib32l2sWRwP3V8sApI0naotwyv1yA+Ecl895MQ=
 v0.12.0-alpha4 h1:/coDhm1r/9fNaOoyssPoOzg4ZFatrn/00fy3J+6KlT4= h1:/NPUSmdWTeKONyjGU/WorQ6BZEPkaAs4yC+LVKL7wy8=
github.com/hashicorp/vault
 v0.11.5 h1:6G3922BuHAxy3icIgSTJiv6GQCqFgdmXBvn3L9bNrZA= h1:KfSyffbKxoVyspOdlaGVjIuwLobi07qD1bAbosPMpP0=
 v0.11.6 h1:+gCdza4h7JiQB3OhQdy1SEp/itCjZASx5pJjKjpxvjM= -
 v1.0.0-beta1 h1:fqH+uqY8IhVqggArFOAHdwQH++ktzXE1p8WMn4psFMQ= -
 v1.0.0-beta2 h1:wZS5jlCwUGWtOtKvQ6JY/8EzYyWLIRPY+qN1uIq9bfo= -
 v1.0.0-rc1 h1:T5ZCugyl03wKpc2blVQ9qU5wxn/oFuRcqMsr3g6kHz4= -
 v1.0.0 h1:TWu6XtmCchkhdsZ2SdMyt5mBKPS/SpTjIQ3Zaubgu2I= -
 v1.0.1 h1:x3hcjkJLd5L4ehPhZcraokFO7dq8MJ3oKvQtrkIiIU8= -
 v1.0.2 h1:CpHnQQKqhquAfC862BiwhksW5Fqhhv0BKlxXpoMlZsA= -
 v1.0.3 h1:8qfP7xbldsLHnTktm1BoxOwlHWLjqr9t7QNbkE4Wbyw= -
 v1.1.0-beta1 h1:96PSGlz5ziWsyWj6Hf1p0TBK1mSGyEG2jvfzKEgNUsk= -
github.com/hawkular/hawkular-client-go
 v0.3.4 h1:s+6IAPZymlvz8R0naNX13zQmF4ua+TWHQZ5xDr3aAyQ= h1:S66kdEKTztNu/GH+yD/+5medwu3yMhSNVaJQuiauls4=
 v0.4.0 h1:jiWZ0AmS0Cpr/fS76bBeFOaSB4lY9ELtvI0LWsZ16do= -
 v0.5.0 h1:mLpRulqBOYyrgO8jKpdR+wDHqKt35nqP6tHa8xUktj4= -
 v0.5.1 h1:T16y8qIBIDHj9Z1vdvx70K8dVb2vO8Ur78p0VMzAaK8= -
 v0.5.2 h1:K4znjMHJ9VndLKTxITuFsc94T6g0aPTLfK0uxjxbRRU= -
 v0.6.0 h1:WCJpLe9e2i4D7nZGbfX2gwdjprSNU+TtkggW9yx88ho= -
 v0.6.1 h1:GIPNWhGSOFpKlse4RR3uQtK9hxX/2pCetZ0gInEmhIw= -
github.com/hmrc/vmware-govcd
 v0.0.1 h1:iuRm76TP4ZvEPZ1RruR71M4I6AmKk8rAX4UGqJL1kCU= h1:SuBoA+q0Lqs2ZSa0rtZFFoRu7MW6TAfBNQPUsR60eB0=
 v0.0.2 h1:hkXDTN2S35vvbzv6vwr3cpRl04H6sC1WJNFCVO8ynS4= -
github.com/hoisie/web
 v0.1.0 h1:oCXxaS9BnWgJZsubx4X+NuOyKPc9y4kB8Eg0jof6SUc= h1:9rKIjxNOF05p21HiYMbaQy+ijn3nHaWi2mV3l/KnoIE=
github.com/howeyc/fsnotify
 v0.8.10 h1:b7UvwW7veKj1Kyu95L89sV596C/LI0e1zisxjkLjyoI= h1:41HzSPxBGeFRQKEEwgh49TRw/nKBsYZ2cF1OzPjSJsA=
 v0.8.11 h1:otVM78zDupDGJDP90DTTUwGu3nv/nfUNxkYLTwiJ42w= -
 v0.8.12 h1:QgKKMaSzmjRFr5btlOw1c8hTxeE1UB8OSSG47SbdDlw= -
 v0.8.13 h1:C9U4sJxJfjwqs2LNyXIoRkSeTf9h3c2IgnQjKAGXXNY= -
 v0.9.0 h1:0gtV5JmOKH4A8SsFxG2BczSeXWWPvcMT0euZt5gDAxY= -
github.com/hpcloud/tail
 v1.0.0 h1:nfCOvKYfkgYP8hkirhJocXT2+zOD8yUNjXaWfTlyFKI= h1:ab1qPbhIpdTxEkNHXyeSf5vhxWSCs/tWer42PpOxQnU=
github.com/huandu/facebook
 v1.7.1 h1:/0+2cI4nHTcQePY+kTPacy9/XfGsVNxmDAcR301UoH4= h1:wJogp9rhXUUjDuhx6ZaR5Eylx3dsJmy0zyFRaPYUq5g=
 v1.8.0 h1:6lQakJHWBaD/MBrbKAelGFMwWBcqZo1B4/2Y8KzRIBA= -
 v1.8.1 h1:MXHiGR+O3+dxzxCHnwrx5K7RreOXWaHaV2Jp8ryQ+g0= -
 v2.0.0+incompatible h1:AzYpEnwEg5EiElbOY+Vi1Rv317B4yRTcBAF0jrzW9Y0= -
 v2.1.0+incompatible h1:y6it6LIu75DATA5SjuWJTFbZteFEhafLDVbqy4tNypc= -
 v2.1.1+incompatible h1:0JXO5taJnV1yF1flECS+bLN3/0J0dabd8jKCJk8LWfA= -
 v2.1.2+incompatible h1:APtODfdBm89WyxkAhP2rZlDvPFJv21fQccS1dRFkTdU= -
 v2.2.0+incompatible h1:glMVeJO7RddaTlV8ex5M99DjBkZSqur9L1+8pdOmHSI= -
 v2.3.0+incompatible h1:+PTxegxsqAiwo1pkMQuQC3OH+zXdrvwHrWgp53kCf6g= -
 v2.3.1+incompatible h1:+F6kUqKx5TifzMg2fXYZFdA/3VVNphdNK8G4PF2ui74= -
github.com/huin/goupnp
 v1.0.0 h1:wg75sLpL6DZqwHQN6E1Cfk6mtfzS45z8OV+ic+DtHRo= h1:n9v9KO1tAxYH82qOn+UTIFQDmx5n1Zxd/ClZDMX7Bnc=
github.com/hybridgroup/gobot
 v1.3.0 h1:yrNNyboEkKhaSZIBvOS7Clzn0iyJ+pO7Jw4a/1aWlc4= h1:ip2cePa6yP93BtAH/QfssMOG49owi6UGhdxjIfYhQC4=
 v1.4.0 h1:3oaMMLMBNS99aoWz2iDKA2g8xhEY4Eo+s9UdMuc4dz0= -
 v1.5.0 h1:IEx5VrXU44rZWeGO/EERUT0MopYBqvPO0RxQSiKBDWg= -
 v1.6.0 h1:GYPCZaArlB99MnGDlnc+l6HPJGINKgD5mGgu8PC7VrM= -
 v1.6.1 h1:pQvUHEQFTtAXUFe2n/jvJz+cS977VcEozzwPirfzHEc= -
 v1.7.0 h1:+burqp8KIRWQtw0q/+DcMZpVs+yJ6HuaTPD0i/hY4Wo= -
 v1.7.1 h1:Klt/y/kUOfrkwK7A3ZTH8/TR9hup1e3B43HevGBQyfk= -
 v1.8.0 h1:PblnXxsxtbOexL2msWFvmb1eLfZAnnnCFSY/kcDzZGE= -
 v1.9.0 h1:iYUxjcubjlKJC9MLjXD0+UfU+raKXBVhlXWyaKIvUWo= -
 v1.12.0 h1:OKKoqn67HSaluBjtLYqW0m0PhvXt51QsYofBxVbOtRY= -
github.com/hyperledger/fabric
 v1.1.0 h1:qEakQEoN5fnw7LDsRnnbC2hDne1takFL3yMedAgObow= h1:tGFAOCT696D3rG0Vofd2dyWYLySHlh0aQjf7Q1HAju0=
 v1.1.1 h1:i8ZTQmgH7WXFPx+Y41krnDdF4h/pGRx+VbQXN9700sI= -
 v1.2.0-rc1 h1:SNjW7/U1PtvfLqGhhndyXh2MtLu0sDo6Rq1hxU7oRgw= -
 v1.2.0 h1:2JzOymSYZaeKv0+DyQr7eMUpc0g3Z2lwjBmMu92tdP4= -
 v1.2.1 h1:tLVSNuzjn8+UYs9NVXo43fnguCC1Ix4Vj3h9iHZrbHc= -
 v1.3.0-rc1 h1:/gNtaIjByh28GrIIjc+KhEs+342HaReFZhKKFZd3TYc= -
 v1.3.0 h1:ijLQL3y0NZcVxfAlMUbeMUpR2Ci+ldiRHim02ohYUwI= -
 v1.4.0-rc1 h1:Cc2eKGKOrr4Brij8/Plg+5Lb5dK0m2WnkDytZNV5GcY= -
 v1.4.0-rc2 h1:4EzlHtIhvPWE5RfDsbnRqwGfv3v1TmWQ+2hX9YLWVTE= -
 v1.4.0 h1:AiOUXysOwh4BmSLZaNzw6ZocZl0uByekPcc+g3PiXH4= -
github.com/imdario/mergo
 v0.3.3 h1:ykJmnl1fiDtSWG6pvkGdccTS4PnsrCN9lPkuzSCA25w= h1:2EnlNZ0deacrJVfApfmtdGgDfMuh/nq6Ok1EcJh5FfA=
 v0.3.4 h1:mKkfHkZWD8dC7WxKx3N9WCF0Y+dLau45704YQmY6H94= -
 v0.3.5 h1:JboBksRwiiAJWvIYJVo46AfV+IAIKZpfrSzVKj42R4Q= -
 v0.3.6 h1:xTNEAn+kxVO7dTZGu0CegyqKZmoWFI0rF8UxjlB2d28= -
 v0.3.7 h1:Y+UAYTZ7gDEuOfhxKWy+dvb5dRQ6rJjFSdX2HZY1/gI= -
github.com/influxdata/influxdb
 v1.6.1 h1:OseoBlzI5ftNI/bczyxSWq6PKRCNEeiXvyWP/wS5fB0= h1:qZna6X/4elxqT3yI9iZYdZrWWdeFOOprn86kgg4+IzY=
 v1.6.2 h1:Cvl0/3n7/T6RkCefitJtEHWKJznmOA+9tT8gVx3vVS0= -
 v1.6.3 h1:TioHM/BpNNH25J89jnL2tk45ww8e2CF+3Q/ih0CMw1I= -
 v1.6.4 h1:K8wPlkrP02HzHTJbbUQQ1CZ2Hw6LtpG4xbNEgnlhMZU= -
 v1.6.5 h1:o/AmF9wd1nq1mpTastR85EVUxL+bwKf83CxfVJHlA1U= -
 v1.7.0 h1:K5PJo6Qla4DtGCY2pjmyMReSvN5DUWosS0cns1O7CnI= -
 v1.7.1 h1:kkc04cz95zGSz1sKSzaP/+7X2r72894aWRCszbimfTc= -
 v1.7.2 h1:+sveWfe1MVK3a7ZkwzB+gJx7th4af+nTANPzIY5L2k4= -
 v1.7.3 h1:9BaicfUiqcYtQfquxpKX8BBlaluDqx7BG1LfMCRsleg= -
 v1.7.4 h1:Ufqfn5xFixUXXj5Fgmhfa9RSke2R2AOvUOXfxgp9SCA= -
github.com/influxdata/telegraf
 v0.1.9 h1:WcrSkAmL37i8TaujLY3JPWD2fWmbIRLq7V/btpklDJY= h1:HIOhVICa+3kYiBmzfDt9LEnDA++FNzRzf9eP0o365us=
 v0.2.0 h1:74mQw5OT1/CIBPmQu5b37hSX4Tdj3FLb11fBwV9puBU= -
 v0.2.1 h1:9Y+YNrtoPs1VdLaau5iq9v0/iplyv70jbSrP8XH6kS0= -
 v0.2.2 h1:Z+SjixSbkF+NV8bXeatE6h5NrJtFabrHnTWcieJMcjo= -
 v0.2.3 h1:wvTgweGUiTG/GkJTb8NzpRXk0uFI86/73bq4lGkn/pQ= -
 v0.2.4 h1:ASVNC9VxMRTMWnSVzzOBWuX6Mht+JO/S9O4xh4DQwac= -
 v0.3.0-beta1 h1:yIZgQ8xVMdqM88SL2IEw2uAlRYest2WguER+M/d3tuQ= -
 v0.3.0-beta2 h1:j8MURhEzLGmzXErhxrcWXjUmiztSqkMDPNRp2C0xLII= -
 v0.10.0 h1:8JjbYOH4YKnZba4BWdiKjj3m8KnTJjE1bgvCBwCaANs= -
 v0.10.1 h1:24IsNJ+yVAvbXiJjIV50taNDPW37b85P6a83LkPDT4c= -
github.com/influxdb/influxdb
 v1.6.1 h1:DP3MPrdfuXLsaRrYBHzJa1M2QeUm/DzBxlfZPc0Xu0E= h1:GpjLgHRqWhDGlPAg7+Rj6NAYuzPojBM8XLG5Ouvvq+Q=
 v1.6.2 h1:CHkLDuW9fDgcckktJMsRAk3zimM0YdBkgKwWJfQhnYM= -
 v1.6.3 h1:6ehQTt1twV88tyoazY6hT8EZX3SZsHG1EIrk5w6UunU= -
 v1.6.4 h1:UWOUH2bH0UPNdcY3YnxWa5SakgEsKqoCaB6egTjeNXI= -
 v1.6.5 h1:UQiAQ8EBtmtorkX82SHRGDAfLUadLqB3+dCRAHCP1tc= -
 v1.7.0 h1:Uz8b9o7xP4r+VF10MUEZpSz+7hUZwu8HvN5IMFsTtvM= -
 v1.7.1 h1:7TcKLR0rwwujjCbGN2t1sEw7TOQ1om6TPpDk/d6wrIg= -
 v1.7.2 h1:XlOFSjNr/j01EdU526tiyBSyvFP3BDGL11COJREttX0= -
 v1.7.3 h1:1/nGRGcqPLnFimlt3aBqr2p1VopehoH5dJ0p4a6ET4w= -
 v1.7.4 h1:4rxgCRO04iRyBmKgE44nM9ebYZ+txkPj2BUWQLx8AyE= -
github.com/intelsdi-x/snap
 v0.8.0-beta h1:zNTsbsM2vaHmo+om4B+yJUd9d3ElLmeTEKZVN75XAPI= h1:PBRZX6nRv10viScFwn4Zn0OY0UX9lIHCTp2+vOW7CpE=
 v0.9.0-beta h1:sT10Fs1QJsBHiiYFdwuV3RfycV6NXF9znens7boy6Ks= -
 v0.10.0-beta h1:rIS/kpBugi7j1nMG2Dip6SpQFMmc1mpIVumRlZbUtxY= -
 v0.11.0-beta h1:M/OYWrV2jHCBTpYikgqSDnIS/sTbBmX1/iDjaE13uSQ= -
 v0.12.0-beta h1:P5FOilgW55l8X/mVmOJQEoEJ7pQcg4dCKxM5vy74NtE= -
 v0.13.0-beta h1:2xLbZ0jYC5Kmf2/Uy2ga3PFfrYnJDMQf7EwTF+0FrbY= -
 v0.14.0-beta h1:MZKCS7p3RE0w0Xxh9MGgGJbe7lui2QP73ArR/SclKDE= -
 v0.15.0-beta h1:DBeME1XfsoWuq2xnhtkDsWzaiEQLR7dgXOFrkvEk73E= -
 v0.16.0-beta h1:iwJCL7CUR7jKiiCWbZHl12diTzh4RtkCjRNfCr2flrI= -
 v0.16.1-beta h1:AtmO3dQzbcEkBnop7HGJI5w379i1QTY5xllsPreJ1MM= -
github.com/ipfs/go-log
 v1.4.0 h1:uBUiTHoQSKYtqzFd1umNcTvi71afSg5avA3c+2tAVNg= h1:AKYS9u+ECLT8t30brTaoVwu3f1FpGx6C0352oI1zQ0Q=
 v1.4.1 h1:T2ifaSU0YQsVOP87eo8NDmav2Coba5uJtsF8DaXG9AU= -
 v1.5.0 h1:4Oauuiq1Pluu3MPGaHpLQ6uOmwINh8eOZUe5SK31+yA= -
 v1.5.1 h1:Pax6qpQ+vqKAZWY8szhGqE2cp3YpGzdfH4kd14r68Vk= -
 v1.5.2 h1:zrE2rC8WYT0HNq65RIHmmFE5uzNhddDSI/azLVNPpZ4= -
 v1.5.3 h1:sZHI8SmguogIAJ10boCjbzumn1hloQF0YM1c0nypQFA= -
 v1.5.4 h1:pXtzlhG8Q8Vj4+6jrEHxvCyn/ceZzr8AWpgGaKltLDE= -
 v1.5.5 h1:QByTOBK2V8X7hY4MQWhv2m7GQarcL35MN9OXvIrmZZA= -
 v1.5.6 h1:ENA6oMqszTahgH9tr9I7DJYI4tcaUk3VfXaDcKW91E8= -
 v1.5.7 h1:8ef7XW41hzAnvVNkK5009/bOA9/MFr7fhdzkfAqvolI= -
github.com/issue9/identicon
 v1.0.0 h1:pnvVMNlWfl+dgTT4rXjG/RFJe7+Ro3fXxcxN92T3Hqk= h1:3AzYqZwDvuBUW/MU1r70SzWmgxZzuj2SyTTUGrZAyT8=
github.com/jackc/pgx
 v2.8.0+incompatible h1:QfXNG0rwud497zBF8IvzN6HfE/BhpBbR9100408ZdgQ= h1:0ZGrqGqkRlliWnWB4zKnWtjbSWbGkVEFm4TeybAXq+I=
 v2.8.1+incompatible h1:DUpuh8MiDYDJtw5Xkkwc4i9JsyriVKGvdq9vUxIY7Hg= -
 v2.9.0+incompatible h1:m72b4cJA7tYXe7z58O5plnF89pQn7qW3Z4pVYND/UYg= -
 v2.10.0+incompatible h1:iTC3HUC9HbeFu+JgxZ5iMj+ZZySfxa3TB3d2hMRQ5z8= -
 v2.11.0+incompatible h1:IgFLUrzrhJj8mxbK44ZYExGVnjtfV4+TOkerb/XERV8= -
 v3.0.0+incompatible h1:ktFg77nZ3QRecqx4KNyuyoxXJCgZMjJWHe8TI1M/jNw= -
 v3.0.1+incompatible h1:svZ2XNsChlQeu4BSOB4ui4X4Bpzss/XY5TKnsS1kkIw= -
 v3.1.0+incompatible h1:G6xyq9OLi10XNimlx3LFe3e+zkYhbYND9nitiMrJx48= -
 v3.2.0+incompatible h1:0Vihzu20St42/UDsvZGdNE6jak7oi/UOeMzwMPHkgFY= -
 v3.3.0+incompatible h1:Wa90/+qsITBAPkAZjiByeIGHFcj3Ztu+VzrrIpHjL90= -
github.com/jackpal/go-nat-pmp
 v1.0.1 h1:i0LektDkO1QlrTm/cSuP+PyBCDnYvjPLGl4LdWEMiaA= h1:QPH045xvCAeXUZOxsnwmrtiCoxIr9eob+4orBN1SBKc=
github.com/jawher/mow.cli
 v1.0.0 h1:jbDHWiSHRJlNixLY9Tg91dd3cx7pi43w/WT0SlTuF5U= h1:5hQj2V8g+qYmLUVWqu4Wuja1pI57M83EChYLVZ0sMKk=
 v1.0.1 h1:H47gmvfl3DlNEHfGmEK+Txm/yxyOzNE86CcSH4DOaec= -
 v1.0.2 h1:CiBs8K6bKCrt6SdVttb+davPTqkBHmaEyhd8gPhcGWU= -
 v1.0.3 h1:Gzeyd6chWE6QOMMcWh/A6mZ/szC5hpkYkqkzj4DakgU= -
 v1.0.4 h1:hKjm95J7foZ2ngT8tGb15Aq9rj751R7IUDjG+5e3cGA= -
github.com/jessevdk/go-flags
 v1.1.0 h1:Geou1o2RJhW9nUu+puVL2ASZMWjfj6+uy97+byGKL98= h1:4FA24M0QyGHXBuZZK/XkWh8h0e1EYbRYJSGM75WSRxI=
 v1.2.0 h1:hzF3gGPUyvR8CkohvbuReyJykgogDQ5bCuNB7LIzgD4= -
 v1.3.0 h1:QmKsgik/Z5fJ11ZtlcA8F+XW9dNybBNFQ1rngF3MmdU= -
 v1.4.0 h1:4IU2WS7AumrZ/40jfhf4QVDMsQwqA7VEHozFRrGARJA= -
github.com/jinzhu/gorm
 v1.9.1 h1:lDSDtsCt5AGGSKTs8AHlSDbbgif4G4+CKJ8ETBDVHTA= h1:Vla75njaFJ8clLU1W44h34PjIkijhjHIYnZxMqCdxqo=
 v1.9.2 h1:lCvgEaqe/HVE+tjAR2mt4HbbHAZsQOv3XAZiEZV37iw= -
github.com/jinzhu/now
 v1.0.0 h1:6WV8LvwPpDhKjo5U9O6b4+xdG/jTXNPwlDme/MTo8Ns= h1:oHTiXerJ20+SfYcrdlBO7rzZRJWGwSTQ0iUY2jI6Gfc=
github.com/jmcvetta/napping
 v2.1.0+incompatible h1:bq99tPkOEqJsTbBkT31te6X38y+3P/FvXVjnxgCqQIs= h1:dlR6SvwNgFr2ASHFGDIO2fhkZM2rU/9B6NB6xUciyv4=
 v2.1.1+incompatible h1:U8vNF2RLEaOZkH8s2jSKZrffOTnnLq5Z0B+8wWDSJIE= -
 v3.0.0+incompatible h1:dunlgXcg5SUnsaStWyjQTxHh9nOsj5/hMM9yiV7U990= -
 v3.0.1+incompatible h1:w0yQ8wSEGS3X+uvpIvOl3YnX9vj2VUhBmz8i6V5rEJg= -
 v3.0.2+incompatible h1:rL9o9EJHwskQkjvBxAveDAYMGS7EcsyDNvZgmUmATK8= -
 v3.0.3+incompatible h1:nVCkMX2j74Ida0AgyO0UtqNa0GJnMdpk8TuF7fKPm5s= -
 v3.0.5+incompatible h1:J3qNDMUA6EhET28WBtCKQn7xamFuAORtDUdoQKK5lSc= -
 v3.1.0+incompatible h1:UIPzx/RBcgct9C8i/cTIL0R0v3O+o/h8PMT87BquqSU= -
 v3.1.1+incompatible h1:2naMYrevDKiVSiPxd1XbUu5m8dtIqmiCMMRPMQUXezc= -
 v3.2.0+incompatible h1:shS22lJu18MtyRV7IqWenMmrRXCjADcUcxONAnp5zxY= -
github.com/jmcvetta/neoism
 v1.1.0 h1:btwSE803uhTVaYN1XShS6udb/xBhcUDt8sMeajRrgp4= h1:oo187spiW9p7dZGW+sMS+rOICd2fqFT9Oc/LdrFz7Kg=
 v1.1.1 h1:zc3iignYGf3AquVhja3MVd/6IJTnBrQJyeBc+7dFf+E= -
 v1.1.2 h1:CpQ9L2b4U9ToxR3QmC0SAmhQBjdp2WFQJtXQjsKHDPQ= -
 v1.2.0 h1:mGkctG9ozbTE255PK7faBEXIm4TE4JRVBJnUqMl2fvY= -
 v1.3.0 h1:H7WeoBC7Q9AtHZJBNYYO98i0xJMx7zoQ099sSRbv2V0= -
 v1.3.1 h1:GCFSl/90OYwEQH5LML/Vy6UlwK4SZ2OIO278UI4K7DE= -
github.com/jmoiron/sqlx
 v1.2.0 h1:41Ip0zITnmWNR/vHV+S4m+VoUivnWY5E4OJfLZjCJMA= h1:1FEQNm3xlJgrMD+FBdI9+xvCksHtbpVBBw5dYhBSsks=
github.com/joho/godotenv
 v1.2.0 h1:vGTvz69FzUFp+X4/bAkb0j5BoLC+9bpqTWY8mjhA9pc= h1:7hK45KPybAkOC6peb+G5yklZfMxEjkZhHbwpqxOKXbg=
 v1.3.0 h1:Zjp+RcGpHhGlrMbJzXTrZZPrWj+1vfm90La1wgB6Bhc= -
github.com/jonboulle/clockwork
 v0.1.0 h1:VKV+ZcuP6l3yW9doeqz6ziZGgcynBVQO+obU0+0hcPo= h1:Ii8DK3G1RaLaWxj9trq07+26W01tbo22gdxWY5EU2bo=
github.com/jrick/logrotate
 v1.0.0 h1:lQ1bL/n9mBNeIXoTUoYRlK4dHuNJVofX9oWqBtPnSzI= h1:LNinyqDIJnpAur+b8yyulnQw/wDuN1+BYKlTRt3OuAQ=
github.com/jroimartin/gocui
 v0.1.0 h1:D96Qxb4ofKJiIwEQmN9+/Z+vARRiuoStBVIYcxYZ28s= h1:7i7bbj99OgFHzo7kB2zPb8pXLqMBSQegY7azfqXMkyY=
 v0.2.0 h1:4UtxqVXvWJRDX0FS13wzV2vafPscxKoslOHEBrN8g9E= -
 v0.3.0 h1:qinwev3/gShLSz/IhB7kMQGO7SbqXFM4TKU3Zv8d8DU= -
 v0.4.0 h1:52jnalstgmc25FmtGcWqa0tcbMEWS6RpFLsOIO+I+E8= -
github.com/json-iterator/go
 v1.1.5 h1:gL2yXlmiIo4+t+y32d4WGwOjKGYcGOuyrg46vadswDE= h1:+SdeFBvtyEkXs7REEP0seUULqWtbJapLOCVDaaPEHmU=
github.com/jteeuwen/go-bindata
 v2.0.1+incompatible h1:BoGQq4qbBRvnem5UFhMX6D69u5ELAjTgH/NtP+JQp+o= h1:JVvhzYOiGBnFSYRyV00iY8q7/0PThjIYav1p9h5dmKs=
 v2.0.2+incompatible h1:8paF6mFmP9mE4JJeo873RMnjrsRew/fjRj5VtZIf8vk= -
 v2.0.3+incompatible h1:TeMRZ8Crz7V9qbqutwNwPj5St9HY0PaiMOEKoEuXoGA= -
 v3.0.1+incompatible h1:azHsZPvV7BHDiQdSIDzLkRoaD/HqfEq6wYiDAa1AhUA= -
 v3.0.2+incompatible h1:G2uCPkgOyO6AkNoTBrYBZSB5aliRvodELG/gSxNM2a0= -
 v3.0.3+incompatible h1:G88+zfeFn6v0NsBXkC5TJOcGbvLyEgJDTNfTPJo48io= -
 v3.0.4+incompatible h1:f18t1uRj1HCVd6z4TEIay0sadVEZ5lJ5AOA+zRsjLts= -
 v3.0.5+incompatible h1:uHsqlPpK6LxFmX6lGK/Y2C+Yz54fOTuI8ccwsQhvmLU= -
 v3.0.6+incompatible h1:bL6Q+9Fu99+Js4uXLfL1nHdUVSVcvmcofTlt1FNdksk= -
 v3.0.7+incompatible h1:91Uy4d9SYVr1kyTJ15wJsog+esAZZl7JmEfTkwmhJts= -
github.com/juju/httprequest
 v1.0.0 h1:WyZyLnwnhjVOGCIw9qwyRLlftIN7jKbjZmN7yvnX/W0= h1:ogcnuLPnijli/VAQT3/CU+2EdSTVsGRfpWTV/VY457I=
 v1.0.1 h1:p7XMlMkx0A8gW6sws2+uHcRr38f9BvF8MQOfKfJihq4= h1:K+CyYVHU/NcfbMpK7YIVobh4U4Fci3EUB2AqIRtl+xs=
 v2.0.0+incompatible h1:+WtiSbRkEwdqKRBi+4JH8PTdNxBa/h8U8RIzdYaMENI= h1:ogcnuLPnijli/VAQT3/CU+2EdSTVsGRfpWTV/VY457I=
github.com/juju/ratelimit
 v1.0.1 h1:+7AIFJVQ0EQgq/K9+0Krm7m530Du7tIz0METWzN0RgY= h1:qapgC/Gy+xNh9UxzV13HGGl/6UXNN+ct+vwSgWNm/qk=
github.com/julienschmidt/httprouter
 v1.0.0 h1:wqU8SDF8HdYSM+My1MRVV+Er1dDqGTH93dqI6J2wL0E= h1:SYymIcj16QtmaHHD7aYtjjsJG7VTCxuUUipMqKk8s4w=
 v1.1.0 h1:7wLdtIiIpzOkC9u6sXOozpBauPdskj3ru4EI5MABq68= -
 v1.2.0 h1:TDTW5Yz1mjftljbcKqRcrYhd4XeOoI98t+9HbQbYf7g= -
github.com/jung-kurt/gofpdf
 v1.0.0 h1:EroSdlP9BOoL5ssLYf3uLJXhCQMMM2fFxCJDKA3RhnA= h1:7Id9E/uU8ce6rXgefFLlgrJj/GYY22cpxn+r32jIOes=
github.com/k0kubun/pp
 v0.0.1 h1:HNWb4RHQrpinW/NN6di95NoAxq8sQUwuuTu/UZB7buk= h1:GWse8YhT0p8pT4ir3ZgBbfZild3tgzSScAn6HmfYukg=
 v1.0.0 h1:ElO1Z7a1Oh30+EgHmYNESZ5fl7GNqy4glrw7PE0oCBI= -
 v1.1.0 h1:/3l1Fra7N1wGzPPGzdnUZ0yxeaEe7c1bs34wBr6VS6o= -
 v1.2.0 h1:QOSAufLQqUuqmtdjYiy7FOb7R5TPxsmhvr6xlxFH6e0= -
 v1.3.0 h1:r9td75hcmetrcVbmsZRjnxcIbI9mhm+/N6iWyG4TWe0= -
 v2.0.0+incompatible h1:N11dO6yXFa+gfbCQqytSQ3z0yB2afnxzr5LiOsjHCiw= -
 v2.0.1+incompatible h1:SN82mMRx22WOazm72mkv4rqU+68AScRQ8i6aSDI1pEE= -
 v2.1.0+incompatible h1:Q1znRhsEpan5denzKeCIOUCyLWQTlopZYpHFHZLu9Ig= -
 v2.2.0+incompatible h1:CJdCM8D10voAW/MDCZwD7b90vL3CUHjq0konQ4uDMZ8= -
 v2.3.0+incompatible h1:EKhKbi34VQDWJtq+zpsKSEhkHHs9w2P8Izbq8IhLVSo= -
github.com/kardianos/service
 v1.0.0 h1:HgQS3mFfOlyntWX8Oke98JcJLqt1DBcHR4kxShpYef0= h1:8CzDhVuCuugtsHyZoTvsOBuvonN/UDBvl0kH+BUxvbo=
github.com/kataras/iris
 v10.4.0+incompatible h1:oVbhEAmn+OXJSWtECeUtVFQTPoSSTl4GTkfylGr+yBE= h1:ki9XPua5SyAJbIxDdsssxevgGrbpBmmvoQmo/A0IodY=
 v10.5.0+incompatible h1:rOhBFECXXhq99uPgmh67pMzrx1j2OO93Y3rwiDgeECo= -
 v10.6.0+incompatible h1:KIXJR6MKynqS9feOURFX+DvT547S127u9IpLJOo1+DE= -
 v10.6.3+incompatible h1:LnUDdsVGZWTD8zSy2IZ5LO3ZClQOWRGPcw0kcldzG0k= -
 v10.6.4+incompatible h1:FJvOvPsr+u6ei8Vbv8/A4v67E8+YMUlRqHegbxVAcVY= -
 v10.6.5+incompatible h1:Vq3qX3JTxM/6RfvMJXZQjTrfvsGVn9DjR7/KveQAbhY= -
 v10.6.6+incompatible h1:SREtCfMe1MQLxZCJmDY7C4dfeFhAec///KC9V5wKQPc= -
 v10.6.7+incompatible h1:K0K2vh7wJTkMCjCbWmuPDP7YpPAOj9L2AtU7AN79pdM= -
 v11.1.0+incompatible h1:hRCQxJTg2+sOmgK7+TUJKy27qL1L+UYWqMbbfjVCKnA= -
 v11.1.1+incompatible h1:c2iRKvKLpTYMXKdVB8YP/+A67NtZFt9kFFy+ZwBhWD0= -
github.com/kellydunn/golang-geo
 v0.4.0 h1:Y7zhjG1d30csXnl/nxHgmXJCSn0+mwVCbU9iFHkqNJY= h1:YYlQPJ+DPEzrHx8kT3oPHC/NjyvCCXE+IuKGKdrjrcU=
 v0.4.1 h1:nzmLSHd08rwAV15WDxV10/Pivn4Xa1UQCG51NHlNsk4= -
 v0.5.0 h1:Fecln6rTLKoeKWM91FndEkyu7tJo9Q/vNyj+jZE9OmA= -
 v0.5.1 h1:wP/9GnjyKmpx5tojzcjrUbhnUP7DEfHxUHsiAmd3xv8= -
 v0.5.2 h1:jk9dLXxhXwhJV38KqODWFMSiuTpuu6VTgSFdNZwh67U= -
 v0.5.3 h1:8jvZQlhTUG4urcyFF4c03oxSW0ProbZR1lKXmJDJl9w= -
 v0.5.4 h1:67+WUWOM+z7Xo5+5e25YUwy+kieY+NZJ68BUloDhPrg= -
 v0.6.0 h1:mLV0mYhczeAz2Af2XTr8DSaKastU8EvUrnbYIXymVBE= -
 v0.6.1 h1:946ajyqiazGjybq1C7fSOM16A4XraEW4djDceo9tmxw= -
 v0.7.0 h1:A5j0/BvNgGwY6Yb6inXQxzYwlPHc6WVZR+MrarZYNNg= -
github.com/kelseyhightower/confd
 v0.9.0 h1:587MZULGb7NXQ5wsi5q/4tsOfDwCvuvKOMvYpJUxgFE= h1:0Q8d8BLyi6OehU2uYJZjOpDnEDouV7LE3bswmPZ7co0=
 v0.10.0 h1:CTpRg4nqjxyJYTW9YtXumtwjMgyOUxjEQnia+vZGhZI= -
 v0.11.0 h1:yGJ+HyMym7VXXHcZ4Bpi7NCNQ5ZVzvZV2N8YKgEcHa0= -
 v0.12.0-alpha2 h1:hPrHdZAozAVktNaWxHHdwKym1YHU6B9/aWpEos5wsCM= -
 v0.12.0-alpha3 h1:i7GV+neZr0KXW+gTB/CMIsBQXx1Sl7DIz5fD6z98BEA= -
 v0.12.0 h1:ETy2fW3MQjnhyioiDzjdF4aXveR3aPFSBONkztVWSKs= -
 v0.13.0 h1:tMyCRWjnhqsOtnVyB7iyTTjFhQxk0dd7kSUge1xibNE= -
 v0.14.0 h1:UXLvu1FPYEvQ4+JKGlsexOcYuFHBf2uQohziJiuhYyM= -
 v0.15.0 h1:PyJAujS2c9yWlbzRlFc3aLAaho8EOHNuXMlyN7VbXmg= -
 v0.16.0 h1:cZTibejsLOJ+FsyXQNmwbnq+arQJooKh1NH/+VqN40o= -
github.com/kelseyhightower/envconfig
 v1.0.0 h1:QbKqpTSIs8IIUPfjr7PmESZaW1CGmXC3VWz6Ijj4Stw= h1:cccZRl6mQpaq41TPp5QxidR+Sa3axMbJDNb//FQX6Gg=
 v1.1.0 h1:4htXR8ameS6KBfrNBoqEgpg0IK2D6rozN9ATOPwRfM0= -
 v1.2.0 h1:ShuWkCxhdgKvpbfMMuCPjAKfdMDS/iClYwdQDByknVk= -
 v1.3.0 h1:IvRS4f2VcIQy6j4ORGIf9145T/AsUB+oY8LyvN8BXNM= -
github.com/kennygrant/sanitize
 v1.2.1 h1:mMPRXg9eiv6j5N1p1ml3CaGnXHfFi0Doqk8Q/KurQAs= h1:LGsjYYtgxbetdg5owWB2mpgUL6e2nfw2eObZ0u0qvak=
 v1.2.2 h1:cVuA2JWFyBOPkrbapO9IfyYFRmOisrtWBNLJirdUgNI= -
 v1.2.3 h1:lMTHgebyLyRtvNyIAnsKyp0CO/zAS8+YmyWRDJ94WBw= -
 v1.2.4 h1:gN25/otpP5vAsO2djbMhF/LQX6R7+O1TB4yv8NzpJ3o= -
github.com/kimor79/gollectd
 v1.0.0 h1:6APoEersLJ2W0iwWj8C7COlienQ8XTWh8np4w0ggI8k= h1:lDxzEAixH34FPZ0nBIpjCu2vR3ZdIKZbbnf2rc+b2ao=
github.com/kisielk/gotool
 v1.0.0 h1:AV2c/EiW3KqPNT9ZKl07ehoAGi4C5/01Cfbblndcapg= h1:XhKaO+MFFWcvkIS/tQcRk01m1F5IRFswLeQ+oQHNcck=
github.com/klauspost/compress
 v1.2.1 h1:z1Ra6IKoPtIeVA8GV0SCQhuo6T4EBjlL9VwonZ8NYBo= h1:RyIbtBH6LamlWaDj8nUwkbUhJ87Yi3uG0guNDohfE1A=
 v1.3.0 h1:kKeUSEWOEaY7o6tEo81HGkvA90Q+qCd1WIkOjr7HgvQ= -
 v1.4.0 h1:8nsMz3tWa9SWWPL60G1V6CUsf4lLjWLTNEtibhe8gh8= -
 v1.4.1 h1:8VMb5+0wMgdBykOV96DwNwKFQ+WTI4pzYURP99CcB9E= -
github.com/klauspost/pgzip
 v1.0.1 h1:J1UvIV7CrOZlEMyfAfnqIzTmoPHEX0ODZsLATb71mkk= h1:Ch1tH69qFZu15pkjo5kYi6mth2Zzwzt50oCQKQE9RUs=
 v1.2.0 h1:SPtjjC68wy5g65KwQS4TcYtm6x/O8H4jSxtKZfhN4s0= -
 v1.2.1 h1:oIPZROsWuPHpOdMVWLuJZXwgjhrW8r1yEX8UqMyeNHM= -
github.com/kr/binarydist
 v0.1.0 h1:6kAoLA9FMMnNGSehX0s1PdjbEaACznAv/W219j2uvyo= h1:DY7S//GCoz1BCd0B0EVrinCKAZN3pXe+MDaIZbXQVgM=
github.com/kr/fs
 v0.1.0 h1:Jskdu9ieNAYnjxsi0LbQp1ulIKZV1LAFgK1tWhpZgl8= h1:FFnZGqtBN9Gxj7eW1uZ42v5BccTP0vu6NEaFoC2HwRg=
github.com/kr/pretty
 v0.1.0 h1:L/CwN0zerZDmRFUapSPitk6f+Q3+0za1rQkzVuMiMFI= h1:dAy3ld7l9f0ibDNOQOHHMYYIIbhfbHSm3C4ZsoJORNo=
github.com/kr/pty
 v1.0.0 h1:jR04h3bskdxb8xt+5B6MoxPwDhMCe0oEgxug4Ca1YSA= h1:pFQYn66WHrOpPYNljwOMqo10TkYh1fy3cYio2l3bCsQ=
 v1.0.1 h1:7MGdCiqXstfzw4RmqNk5bW7ZuKFFZT9SkCUUGc9SWDM= -
 v1.1.0 h1:T6yjk/ScS0c7co+9lVJEIuPfU3FldAB16dvESPaoDoc= -
 v1.1.1 h1:VkoXIwSboBpnk99O/KFauAEILuNHv5DVFKZMBN/gUgw= -
 v1.1.2 h1:Q7kfkJVHag8Gix8Z5+eTo09NFHV8MXL9K66sv9qDaVI= -
 v1.1.3 h1:/Um6a/ZmD5tF7peoOJ5oN5KMQ0DrGVQSXLNwyckutPk= -
github.com/kr/text
 v0.1.0 h1:45sCR5RtlFHMR4UwH9sdQ5TC8v0qDQCHnXt+kaKSTVE= h1:4Jbv+DJW3UT/LiOwJeYQe1efqtUx/iVham/4vfdArNI=
github.com/kyokomi/emoji
 v0.0.1 h1:7VhY4Nk1cR70WvgkdsxSORwHlW7xB7N/D6qajzkbrPQ= h1:mZ6aGCD7yk8j6QY6KICwnZ2pxoszVseX1DNoGtU2tBA=
 v0.0.2 h1:iU96v4m7YzOpCq6G4OlN4PTCbCtEpQm6KjObXrjq+J4= -
 v1.5.1 h1:qp9dub1mW7C4MlvoRENH6EAENb9skEFOvIEbp1Waj38= -
 v2.0.0+incompatible h1:ftqSD1PzBkSr3I4/0ZZ7Cluaxv6sNi8CkhYE56Son6A= -
 v2.0.1+incompatible h1:FhAFYCGqEJSsh4g/D+QAhousxF8ajJ0LdUf0Xjm+gKc= -
 v2.1.0+incompatible h1:+DYU2RgpI6OHG4oQkM5KlqD3Wd3UPEsX8jamTo1Mp6o= -
github.com/labstack/echo
 v3.2.5+incompatible h1:gRoNZHgcubt9mWO5vW5MDlADiQt8GVHH2EJDvImDYAw= h1:0INS7j/VjnFxD4E2wkz67b8cVwCLbBmJyDaka6Cmk1s=
 v3.2.6+incompatible h1:28U/uXFFKBIP+VQIqq641LuYhMS7Br9ZlwEXERaohCc= -
 v3.3.0+incompatible h1:gv5qUy/eIlbc+jM13CE6O2ezbkN1e6S7y1XKSzHZ0dg= -
 v3.3.1+incompatible h1:bjU/fJR5CPNijPkd4k2U6tKVVc+R6T/xXZQd03G75sQ= -
 v3.3.2+incompatible h1:w0jtWDcIVMCu2IpCvcj6WbJ+xg4D+0Py/uin+zN9j5k= -
 v3.3.3+incompatible h1:ukZRJMwpJrfcFhdiyow9V1/kPCvaFAaGi8a8uwU/8xE= -
 v3.3.4+incompatible h1:83oKvzg2Fa6PN1pQfsUxlfR4MiSjsFzyxbEXKzsE0cQ= -
 v3.3.5-retag+incompatible h1:BpR2lD7yrxczR2+x7mEvd5ykyUHtj/PCqq6CKMrahWw= -
 v3.3.5+incompatible h1:9PfxPUmasKzeJor9uQTaXLT6WUG/r+vSTmvXxvv3JO4= -
 v3.3.10+incompatible h1:pGRcYk231ExFAyoAjAfD85kQzRJCRI8bbnE7CX5OEgg= -
github.com/labstack/gommon
 v0.0.1 h1:DrrX8TJy2HZN5dfLJA9c3QQEAi+Qqu93K2TuAJ+STfg= h1:/tj9csK2iPSBvn+3NLM9e52usepMtrd5ilFYA+wQNJ4=
 v0.0.2 h1:Uk+AyGUXbI6Im8xBUXLzeMcswMYDeW2A+sbqoSY0EUQ= -
 v0.0.3 h1:BpbH1uyyKVcVhVMbiodV+vTQv2OHatlZy/SV9mlFyJM= -
 v0.1.0 h1:8M8zN5f8sUWXU5mj0xAR+C+t0zjUKmkx5sXyih93zsc= -
 v0.2.0 h1:CmGHRiCIUd9wa6FfDsQ6zCKFdqVXIr6MMFipRmkLdh4= -
 v0.2.1 h1:C+I4NYknueQncqKYZQ34kHsLZJVeB5KwPUhnO0nmbpU= -
 v0.2.7 h1:2qOPq/twXDrQ6ooBGrn3mrmVOC+biLlatwgIu8lbzRM= -
 v0.2.8 h1:JvRqmeZcfrHC5u6uVleB4NxxNbzx6gpbJiQknDbKQu0= -
github.com/lann/squirrel
 v1.1.0 h1:kaqNksjVyUngjYKceeSGXPdJFnCGmILgnKJxIESk52g= h1:yaPeOnPG5ZRwL9oKdTsO/prlkPbXWZlRVMQ/gGlzIuA=
github.com/lib/pq
 v1.0.0 h1:X5PMW56eZitiTeO7tKzZxFCSpbFZJtkMMooicw2us9A= h1:5WUZQaWbwv1U+lTReE5YruASi9Al49XbQIvNi/34Woo=
github.com/libp2p/go-libp2p-peer
 v2.3.0+incompatible h1:ba3gtfQixgq1Rytl2muUpZNuKId2rehuCQCW+58PKrc= h1:fS2eFKRO1IomwBAf+SuE8P1XOT/AAiqSgVPNIFA7Jc0=
 v2.3.1+incompatible h1:4tutj9cUPVwn97AaxDoWjLr+acZ9G2FUltnCYEVipZU= -
 v2.3.2+incompatible h1:PnOl+l5B8Bbs+CbZJI0A7XCwIKKRJYvDNYYH+H7WkgI= -
 v2.3.3+incompatible h1:jgrgSO6tgm8O/o03Jq/SBoB+n2eKg2TBZsT48G/0I7Q= -
 v2.3.4+incompatible h1:YgsaA3a5nL0NANtXzhIpQSbyFXmVuxbYMgXl9TcG/uA= -
 v2.3.5+incompatible h1:zgwcWNo8+/Tuy7l7V+HT63bgbOZmXGGcAf1viMhK0pw= -
 v2.3.6+incompatible h1:M7IFxOGTGbNa8BPSvr1nqCf6is3Nb9VhSp1hkL7Aglk= -
 v2.3.7+incompatible h1:AhkW2YVjTfC8u7lnQqT/jZdTt3p2eyShY0dyF6Jyepc= -
 v2.3.8+incompatible h1:ZMYXJdcX/38UBDjrl5BEYRGRgDkc+r+oMSfysyPM/MQ= -
 v2.4.0+incompatible h1:1THIuO/h7GuITklYS7RgGCyoVl8aP9XH4NcokcdhDZc= -
github.com/looplab/fsm
 v0.1.0 h1:Qte7Zdn/5hBNbXzP7yxVU4OIFHWXBovyTT2LaBTyC20= h1:m2VaOfDHxqXBBMgc26m6yUOwkFn8H2AlJDE+jd/uafI=
github.com/lucas-clemente/quic-go
 v0.5.0 h1:Q2eV3gF1uoPxxVx1+/R4qnKPlilFWIWjasC85oHZ2QY= h1:wuD+2XqEx8G9jtwx5ou2BEYBsE+whgQmlj0Vz/77PrY=
 v0.6.0 h1:tPo/vfiSbHVSZsSvHNd9MDlUG6A6etSmlIluGcPLWvQ= -
 v0.7.0 h1:yc+wZPOA6Lut71NIDqD8XLT+OZKPlseu8V1+jL/LxJ8= -
 v0.8.0 h1:35pBTXknOhbz5V+cgXuBDFrmxcgHW6ESbVyoGuXhgFM= -
 v0.9.0 h1:yQlTpNitV317oQntYwe1I3wtuI+8MFYLhbB6qVqoo/k= -
 v0.10.0-no-integrationtests h1:K9YrKQNB9OSMOkX+PhQ30YzuYpgt7FfgFYITvIPrk7o= -
 v0.10.0 h1:xEF+pSHYAOcu+U10Meunf+DTtc8vhQDRqlA0BJ6hufc= -
 v0.10.1-no-integrationtests h1:iiJHgXPEepBV40kt8qRWycOxrUGS9XA6RX/ShO9pKzw= -
 v0.10.1 h1:ipcMmYP9RT+b1YytOKGUY1qndxPGOczVEQkAVz3CZrs= -
github.com/lunny/tango
 v0.3.1 h1:XYpre/m3wZ7KMbT9XGNChMkW2d3rQgtyLsMvYHEec/k= h1:UYYNJtWM9e9sa0/tuwCZrgmWQu6H5S9/dF2VvgVLchA=
 v0.4.0 h1:SQv1/uK03FQ/bhCwDwMSxFmryqJXbPkiOY+g2wO+4ZM= -
 v0.4.2 h1:VNInWo2t9b2wgb3p+XBSH6O4k/EeR6EjddX6sR/Cerg= -
 v0.4.3 h1:OIhU6+8VwCmwx3xe4wMU/cPdHFWrZvzDSi5WQl6lIzE= -
 v0.4.4 h1:/A5sDVFif0kNGVvPOJjCrHh09cjuhroTKjHNmggCUl4= -
 v0.4.5 h1:dMjlJvRYn6HmXCCllZV/ZPQJsn+sW1vm3s4LZ0zPZjk= -
 v0.4.6 h1:B2tKMgVvqjGs0ZbdLTiKd/P3A+FlbGcRb9ShLDBHHi4= -
 v0.4.8 h1:HSZYRoq9jwwdQBUs08Cu9iJjFb7c9C6wRlg3zqO8FDE= -
 v0.5.0 h1:ThYFWPh6tVFjf6s6jKcTe86DadgWRjF4PoDeaSA4zhY= -
 v0.5.5 h1:A2m5hRiRyVZtbR9BkrgB8Bswi8L1s97vQglM7DfAtt8= -
github.com/lunny/xorm
 v0.2.1 h1:BxnqMdmntk6PI5NpEJ3EYMzIZAcDmKp+RDiDbGL2Jgw= h1:dF05xwM+wPY9LVFW4VulsTHZAW1TZZwISDk2SGCd9Ao=
 v0.2.2 h1:fOyQ61XXRdlWVl06aKmmGNhfXk6MLSo/LXP85gcLSxM= -
 v0.2.3 h1:nE+XvUokH3hqpR8zgULm5+9SHNK2n12V2Q8dAE4IsB4= -
 v0.3.1 h1:cd1S/u9UswQxuT3xykQoSp+RXJADX3nokTv9A0pm4Aw= -
github.com/magiconair/properties
 v1.5.6 h1:C09pD/B3qhKWsYNwGcgotqsPMZPlF5YQoJ2bfNsrSwY= h1:PppfXfuXeibc/6YijjN8zIbojt8czPbwD3XqdrwzmxQ=
 v1.6.0 h1:HtUzhdcKHtHmlTXBzD3oM3pyFYFdR+fzzgPwJAzcAes= -
 v1.7.0 h1:8Oygzd+nhiSWfp2XOwattyfEt36Q/F/99zQJy4axKdg= -
 v1.7.1 h1:y1yaj5GXa4mhCIWT/DKBLFWvT01Wy+nbRRri/vLvq5o= -
 v1.7.2 h1:de14gtQSJmD380aERiT7g/BPsA0iDosrVRaU+EhZAio= -
 v1.7.3 h1:6AOjgCKyZFMG/1yfReDPDz3CJZPxnYk7DGmj2HtyF24= -
 v1.7.4 h1:UVo0TkHGd4lQSN1dVDzs9URCIgReuSIcCXpAVB9nZ80= -
 v1.7.5 h1:yaOSImqiEHqVrOCCwSkX72kfRXuEEzkzOU4SevKv24M= -
 v1.7.6 h1:U+1DqNen04MdEPgFiIwdOUiqZ8qPa37xgogX/sd3+54= -
 v1.8.0 h1:LLgXmsheXeRoUOBOjtwPQCWIYqM/LU1ayDtDePerRcY= -
github.com/mailgun/mailgun-go
 v1.0.0 h1:fUIiAHYwFwFziTBNtIssejCHOXPvdEJaseKg2JSPfG0= h1:NWTyU+O4aczg/nsGhQnvHL6v2n5Gy6Sv5tNDVvC6FbU=
 v1.0.1 h1:qjObNIt6loI1P6Rfmxn9tJz1xTvrvy6Ed39in8tqhcQ= -
 v1.0.2 h1:YrlCHTyA+CAoxWDPihdY0jY1NxsdXE0AvIAe/Py38vk= -
 v1.1.0 h1:nWaeVwj8F+KFtyMHxjhlQUt1zBK0OhbHjLeVA8u2o8I= -
 v1.1.1 h1:mjMcm4qz+SbjAYbGJ6DKROViKtO5S0YjpuOUxQfdr2A= -
 v2.0.0+incompatible h1:0FoRHWwMUctnd8KIR3vtZbqdfjpIMxOZgcSa51s8F8o= -
github.com/markbates/goth
 v1.45.7 h1:Zv0cMyxjdPcJqWlVcuVTSCLrnkxi7Fe3pPCKJYzZVtE= h1:ERjpUjiHOcJUNTBjgUhpKzkay5qNGcMdjRHYOIpF5Uk=
 v1.45.8 h1:Msgjd4ulbjEPfjKCW3Vlks+HfxmHhNF6e1pmziNFyHE= h1:vOvF9V44wREdp5qcvaht/jnZcNFgpFd0ljcErYRECcM=
 v1.45.9 h1:c1y5Tw6hYkrzrtJTfHTwEcA+0G4DgS6/Pp1tsOae5Tw= -
 v1.46.0 h1:UCPxdYdkmG/SpA/QQY6cLzRjHdvwdRVAoFHgxyPsFvg= -
 v1.46.1 h1:SUPPgc+fCOs33qo4J3IbU+5Y8hmi4khnydm6AgH38LQ= -
 v1.47.0 h1:LWduf3MLIKHqj4NqUxxCm2WQK3cIZNJwKbBaquP3rJw= h1:ehX90HlYXaqutZBYjI7wGP2PT5g06Yd1sdzKSLJfveY=
 v1.47.1 h1:NTC7OB+G5GiNGJvxDndd6koHbiPgYnaA6Tk4zZJ8+dU= -
 v1.47.2 h1:SWjRkpI8PsSAT1cxa0DwBpki2WmkeAbBIOl6IVlg1YY= -
 v1.48.0 h1:5udgvaLO9qyQLAUGT5SJW8WYB+ahQgN3TISjzONrAUE= h1:zZmAw0Es0Dpm7TT/4AdN14QrkiWLMrrU9Xei1o+/mdA=
 v1.49.0 h1:qQ4Ti4WaqAxNAggOC+4s5M85sMVfMJwQn/Xkp73wfgI= -
github.com/mattermost/platform
 v5.7.0+incompatible h1:Ns4MWYHDtL0HU5Ld75ILHjT+xiaoLWKq50hhY8jNh7c= h1:HjGKtkQNu3HXTOykPMQckMnH11WHvNvQqDBNnVXVbfM=
 v5.7.1-rc1+incompatible h1:5Ab3GAKBIEDuhLQBGMboE1JpCs34MeE+HAs8quyYhp0= -
 v5.7.1+incompatible h1:ClP1OFZS2iOrmWqDHAY0XfhrZw3v+5cYD6/NU+h3JVY= -
 v5.7.2-rc1+incompatible h1:kIDoPGlGUb6192InXazldyWe6BzE56IFtWjQ7XKuPyk= -
 v5.7.2+incompatible h1:YvzZ2yHBHqOhs4s03ZojoQCuK3l2j6KsO6vhKJ0aQXs= -
 v5.8.0-rc1+incompatible h1:pI6DZHHnToetB7BWPoWRBxspyn2JRO6gYbaGe5FvehU= -
 v5.8.0-rc2+incompatible h1:2jvP9rPI5oYdfrBF3pFYjLRtcfixP15tods6ac+3A/k= -
 v5.8.0-rc3+incompatible h1:ZhpwGSfOB1qVGnB9oQVZSG915OuG93AornRTXXSVVHY= -
 v5.8.0-rc4+incompatible h1:qYSA6PgZCR4ZYrl3nxNI6dSj3gy6payvG2KnB7D4KMg= -
 v5.8.0+incompatible h1:6dRdGAaaavG/vaKq7uxh5o7lSf7pdVD6N1ONqbcAnm0= -
github.com/mattes/migrate
 v1.3.1 h1:kaUHjsvvmhGIkt9WVaEn36Z08+CaHAcXWcnZj3JpSaY= h1:LJcqgpj1jQoxv3m2VXd3drv0suK5CbN/RCX7MXwgnVI=
 v1.3.2 h1:cZkxdp0Zhlk3nycFyk6gmVuvyGcwjta5k2uTL/hJK6M= -
 v3.0.0-prev0+incompatible h1:7XGrukAv7JA7A9W9T+2qNP9Scf8CKv0Xio/y65oWM8I= -
 v3.0.0-prev1+incompatible h1:a6DTJl+bc8UkJqCPyNugst6+n6Vb+jOtE4QecINzxs8= -
 v3.0.0-prev2+incompatible h1:q3sabHVLuqNGNCf3aHpn+C5Hc9F6/wlbR9ZqRhC97as= -
 v3.0.0-rc0+incompatible h1:pymKM+uohHrm5ThaTLQmmmE4i/MUbMtgN5lq1uIFkp4= -
 v3.0.0-rc1+incompatible h1:Oar9mtZ7NZwmcfnR0BPjcScKBlIPkoPG0DHT5+tDeXM= -
 v3.0.0-rc2+incompatible h1:lRefEWXzEnbUQoF1qb2laFNxF791JU4iaecZKZLk2CI= -
 v3.0.0+incompatible h1:/2dVkDQPZf83+dRtTbTLzy0xfq6VwJvLD356y3kpr4Y= -
 v3.0.1+incompatible h1:PhAZP82Vqejw8JZLF4U5UkLGzEVaCnbtJpB6DONcDow= -
github.com/mattn/go-colorable
 v0.0.1 h1:hdWOZaNIZTn/k1zbq9P7SuyHbKLKvGs0vVL1qQvj5sA= h1:9vuHe8Xs5qXnSaW/c/ABM9alt+Vo+STaOChaDxuIBZU=
 v0.0.2 h1:L6EoMpDAuxkwGgsSzzMx2FFJE1/FRISbx4TcsaOwZ0k= -
 v0.0.3 h1:i8hcMpE5C7qWI8b1++M/4RzoJ6pA5TEjloaAm2oCRP4= -
 v0.0.4 h1:g1xCGgHkuL4Ec7in+UFw1PUz/35aWcj1NnEFW0wiiQE= -
 v0.0.5 h1:X1IeP+MaFWC+vpbhw3y426rQftzXSj+N7eJFnBEMBfE= -
 v0.0.6 h1:jGqlOoCjqVR4hfTO9H1qrR2xi0xZNYmX2T1xlw7P79c= -
 v0.0.7 h1:zh4kz16dcPG+l666m12h0+dO2HGnQ1ngy7crMErE2UU= -
 v0.0.8 h1:KatiXbcoFpoKmM5pL0yhug+tx/POfZO+0aVsuGhUhgo= -
 v0.0.9 h1:UVL0vNpWh04HeJXV0KLcaT7r06gOH2l4OW6ddYRUIY4= -
 v0.1.0 h1:v2XXALHHh6zHfYTJ+cSkwtyffnaOyR1MXaA91mTrb8o= -
github.com/mattn/go-isatty
 v0.0.1 h1:CUzGjDU3sTrdhXv9lYvVXq04SYzwkvgm46qjB0T7jGM= h1:M+lRXTBqGeGNdLjl/ufCoiOlB5xdOkqRJdNxMWT7Zi4=
 v0.0.2 h1:F+DnWktyadxnOrohKLNUC9/GjFii5RJgY4GFG6ilggw= -
 v0.0.3 h1:ns/ykhmWi7G9O+8a448SecJU3nSMBXJfqQkl0upE1jI= -
 v0.0.4 h1:bnP0vzxcAdeI1zdubAl5PjU6zsERjGZb7raWodagDYs= -
github.com/mattn/go-runewidth
 v0.0.1 h1:+EiaBVXhogb1Klb4tRJ7hYnuGK6PkKOZlK04D/GMOqk= h1:LwmH8dsx7+W8Uxz3IHJYH5QSwggIsqBzpuz5H//U1FU=
 v0.0.2 h1:UnlwIPBGaTZfPQ6T1IGzPI0EkYAQmT9fAEJ/poFC63o= -
 v0.0.3 h1:a+kO+98RDGEfo6asOGMmpodZq4FNtnGP54yps8BzLR4= -
 v0.0.4 h1:2BvfKmzob6Bmd4YsL0zygOqfdFnK7GR4QL06Do4/p7Y= -
github.com/mattn/go-shellwords
 v1.0.0 h1:xlTU5yhz4gG9QkPtaLZeDXCbsEkX+Ve2rxmVmG5PdSs= h1:3xCvwCdWdlDJUrvuMn7Wuy9eWs4pE8vqg+NOMyg4B2o=
 v1.0.1 h1:2/mQs/EosKUge1MHnAavnrNwa0wLnWDjG4dTYMGf/kI= -
 v1.0.2 h1:5FJ7APbaUYdUTxxP/XXltfy/mICrGqugUEClfnj+D3Y= -
 v1.0.3 h1:K/VxK7SZ+cvuPgFSLKi5QPI9Vr/ipOf4C1gN+ntueUk= -
github.com/mattn/go-sqlite3
 v1.1.0 h1:uggQm4+cc4c0du7NMV5XaXTnHRd0Zx9KMCT6csVT6ZI= h1:FPy6KqzDD04eiIsT53CuJW3U88zkxoIYsOqkbpncsNc=
 v1.2.0 h1:h2FYSp18EBpSL2XOLiU2jIvYcJpx5NxGmT2EFlaUesw= -
 v1.3.0 h1:NDrHgbss6o+4wsrCRAJbPLDTrVdsowaYSkhB16RHZy8= -
 v1.4.0 h1:uBR791wsGR0MwOn2qBSbln4BRuv1/sN7jP0wQ8D6RVs= -
 v1.5.0 h1:cD1JkMVOQgN+75Jni3VEkSwLkElfpfS194KbtOH9jX8= -
 v1.6.0 h1:TDwTWbeII+88Qy55nWlof0DclgAtI4LqGujkYMzmQII= -
 v1.7.0 h1:CiYZ8slwBLIMkDbDJCF+Zd2M8bZ1Gz02TMsm1V33Lk0= -
 v1.8.0 h1:n4Yp7m+83/fCZWiO7nnf6WZAB41luGNFae+GMQPPe50= -
 v1.9.0 h1:pDRiWfl+++eC2FEFRy6jXmQlvp4Yh3z1MJKg4UeYM/4= -
 v1.10.0 h1:jbhqpg7tQe4SupckyijYiy0mJJ/pRyHvXf7JdWK860o= -
github.com/mattn/go-zglob
 v0.0.1 h1:xsEx/XUoVlI6yXjqBK062zYhRTZltCNmYPx6v+8DNaY= h1:9fxibJccNxU2cnpIKLRRFA7zX7qhkJIQWBb449FYHOo=
github.com/matttproud/golang_protobuf_extensions
 v1.0.0 h1:YNOwxxSJzSUARoD9KRZLzM9Y858MNGCOACTvCW9TSAc= h1:D8He9yQNgCq6Z5Ld7szi9bcBfOoFv/3dc6xSMkL2PC0=
 v1.0.1 h1:4hp9jkHxhMHkqkrB3Ix0jegS5sx/RkqARlsWZ6pIwiU= -
github.com/mesos/mesos-go
 v0.0.1 h1:TZ7qdkDAcYu40pTNt/2LscEd3rOnaQfBC3UFlCPqH1g= h1:kPYCMQ9gsOXVAle1OsoY4I1+9kPu8GHkf88aV59fDr4=
 v0.0.2 h1:gLQdBLR7dVXf6TRWpUrrtJcjdTL/exPAsWShUmEyrZc= -
 v0.0.3 h1:3saQcA0BT72yo68KgNsCP3q1bc0GZmMTi5f6sVPUlUM= -
 v0.0.4 h1:hlWy3Efmn8uF6D0hYRk8BZ2tId5azBTkx3AjOtg4ae8= -
 v0.0.5 h1:prwQ9OgQHK9WynDEXU71qxcdP0QK3s5/jIzSfz5J/6g= -
 v0.0.6 h1:OJZFaPVAjfaT1+wrKmB7LIwv6WOXMwANHcKXaPmzPEA= -
 v0.0.7 h1:jQuDrBofpRbVZ+cq/VpMipKQFHNMqXc2arGSR6sVY30= -
 v0.0.8 h1:hiAUHba+ycyZLxDiBUqKs91a0LbHZaAca988LzN19xM= -
github.com/mgutz/str
 v1.1.0 h1:nwAiHNDx58Ps8MY+1bpZArKVcJcKUqdTw8JptE/pdP8= h1:w1v0ofgLaJdoD0HpQ3fycxKD1WtxpjSo151pK/31q6w=
 v1.2.0 h1:4IzWSdIz9qPQWLfKZ0rJcV0jcUDpxvP4JVZ4GXQyvSw= -
github.com/mholt/archiver
 v1.1.1 h1:H/dlJxc+8p3XZhe0BW/t6PDuk3OsV7pl+10ws+1oGHQ= h1:Dh2dOXnSdiLxRiPoVfIr/fI1TwETms9B8CTWfeh7ROU=
 v1.1.2 h1:xukR55YIrnhDHp10lrNtRSsAK5THpWrOCuviweNSBw4= -
 v2.0.0+incompatible h1:KGdPVnP9sU8V6bvSr9v3B97yxKskJUm8U3okpC8WYmk= -
 v2.1.0+incompatible h1:1ivm7KAHPtPere1YDOdrY6xGdbMNGRWThZbYh5lWZT0= -
 v3.0.0+incompatible h1:iylGhvHjW96mpipixw7zQq/DitJiHcY8/a7EOBWDx54= -
 v3.0.1+incompatible h1:y4NDwY7mZ+KILVeb8+E5glIkleoOomYY40qomAvp/Jw= -
 v3.1.0+incompatible h1:S1rFZ7umHtN6cG+6cusrfoXTMPqp6u/R89iKxBYJd4w= -
 v3.1.1+incompatible h1:1dCVxuqs0dJseYEhi5pl7MYPH9zDa1wBi7mF09cbNkU= -
github.com/mholt/binding
 v0.1.0 h1:rIR/c66hEh/SuxbVeRzF9IJP7f0OsRtDGljkjxcyKns= h1:fdTUY9qwc5FrbACIVqskE2Yjb3O/Kt9cPJ6TauRzkjw=
 v0.2.0 h1:YiMIiIJJE2EuJo+lNtrLtcIKgdRQHA7E3H2JBM4xmsk= -
 v0.3.0 h1:gGLnN9XAbyi5st4t1vDfmhOkHUbMwUGegVdnAnCRYvk= -
github.com/mholt/caddy
 v0.10.10 h1:8EawDo3o02ZMQrNtN8t2ygWBGgOas//r/ckrnd0HpxY= h1:Wb1PlT4DAYSqOEd03MsqkdkXnTxA8v9pKjdpxbqM1kY=
 v0.10.11 h1:s8X+R8DuBbrrMuUTcWSxlDe567B0s5EDmiDBKSYsioY= -
 v0.10.12 h1:B/3/91ABQih4n9Vm9RLMzw9L26lKJY4IqVltg9erpB4= -
 v0.10.13 h1:j7YQBPn9WNhvcAkXvNNm1Z6DRYBVX2wZepCbHnvIdYc= -
 v0.10.14 h1:DQbK1E8/gOmRdAymJagUMFuW/8L7+D/XtPifpBw5/HM= -
 v0.11.0 h1:cuhEyR7So/SBBRiAaiRBe9BoccDu6uveIPuM9FMMavg= -
 v0.11.1 h1:oNfejqftVesLoFxw53Gh17aBPNbTxQ9xJw1pn4IiAPk= -
 v0.11.2 h1:9EPjXWMDwpQlmSb57WA4/GBmTbYewJ04e0qWX+nbHXE= -
 v0.11.3 h1:znIxClGweLx4sX0MxNEb/4QjGmOYAiWZAHFR5Qixr70= -
 v0.11.4 h1:he7Ej5Jf9CXjETtfQQBr5KJ1b5ZWdPaBOJjiQs6LAIk= -
github.com/micro/cli
 v0.1.0 h1:5DT+QdbAPPQvB3gYTgwze7tFO1m+7DU1sz9XfQczbsc= h1:jRT9gmfVKWSS6pkKcXQ8YhUyj6bzwxK8Fp5b0Y7qNnk=
github.com/micro/go-micro
 v0.20.0 h1:SBxHxy37fbnyo+jLeP+4l+BLa8eh+rOckgB1iwWDTZI= h1:3z3lfMkNU9Sr1L/CxL++8pVJmQapRo0N6kNjwYDtOVs=
 v0.21.0 h1:lmvZcLMokoGExbLY9ug/VcvMyjOnEqMeSj9IFTf6BNA= -
 v0.22.0 h1:b2gMeqa5YHmSIBS7WEGro+ZxooF7YmIsEWs2UUzo2VY= -
 v0.22.1 h1:U4gZPakGOhG1ypJw+kgQ1+O0JkDePntGjcdc7WSoR0E= -
 v0.23.0 h1:1/nCg4j2IogYRXHu0BJOi0qltbGaEXcq0UCrumx3YcM= -
 v0.24.0 h1:NxyVp/lQOqngxUrYZk/7rw+T3SE0f28JHBhAJqVZI94= h1:G/2AWGXaoz2RoiT8xNNzE9Jn46MAI3GiRvSw6QsCDwI=
 v0.24.1 h1:rNAAPrIdlO5gcb+F/4eUMr9BGdoSu8mkKSjH3kmOggg= -
 v0.25.0 h1:1PZWjxV216ZTU1fRl8s+nkY3eazpnkXYZA9GneRdPJw= -
 v0.26.0 h1:WfLYCJWAuTY1f8qSu33MwWx4S7JctMVQAuUwWnxOLiE= h1:CweCFO/pq8dCSIOdzVZ4ooIpUrKlyJ0AcFB269M7PgU=
 v0.26.1 h1:MJIwdZE5Bi+ptayIwV2f26rjBQdcAj4EUzXhrTSc9wA= h1:Jgc5gPEmDiG1TWE5Qnzzx5qyXnU9VTXKT1FkXkfvt8g=
github.com/microcosm-cc/bluemonday
 v1.0.0 h1:dr58SIfmOwOVr+m4Ye1xLWv8Dk9OFwXAtYnbJSmJ65k= h1:hsXNsILzKxV+sX77C5b8FSuKF00vh2OMYv+xgHpAMF4=
 v1.0.1 h1:SIYunPjnlXcW+gVfvm0IlSeR5U3WZUOLfVmqg85Go44= -
 v1.0.2 h1:5lPfLTTAvAbtS0VqT+94yOtFnGfUWYyx0+iToC3Os3s= h1:iVP4YcDBq+n/5fb23BhYFvIMq/leAFZyRl6bYmGDlGc=
github.com/miekg/dns
 v1.0.11 h1:spHZYYgVtr7q1o9F/AtZuJTVCTml7JkuLSccay9QdP4= h1:W1PPwlIAgtquWBMBEV9nkV9Cazfe8ScdGz/Lj7v3Nrg=
 v1.0.12 h1:814rTNaw7Q7pGncpSEDT06YS8rdGmpUEnKgpQzctJsk= -
 v1.0.13 h1:Y72t3Ody/fSEkLQOC49kG0ALF7b8ax2TouzPFgIT40E= -
 v1.0.14 h1:9jZdLNd/P4+SfEJ0TNyxYpsK8N4GtfylBLqtbYN1sbA= -
 v1.0.15 h1:9+UupePBQCG6zf1q/bGmTO1vumoG13jsrbWOSX1W6Tw= -
 v1.1.0 h1:yv9O9RJbvVFkvW8PKYqp4x7HQkc5RWwmUY/L8MdUaIg= -
 v1.1.1 h1:DVkblRdiScEnEr0LR9nTnEQqHYycjkXW9bOjd+2EL2o= -
 v1.1.2 h1:Y/HbdlkFiiRU3Njr3hRk0KFKinYX90x7wtQMZvxShJo= -
 v1.1.3 h1:1g0r1IvskvgL8rR+AcHzUA+oFmGcQlaIm4IqakufeMM= -
 v1.1.4 h1:rCMZsU2ScVSYcAsOXgmC6+AKOK+6pmQTOcw03nfwYV0= -
github.com/miekg/mmark
 v1.3.1 h1:SlJjKN8nYTIX0y/LUGVamAVSUDS4e5HxOBa7AjyUVxg= h1:w7r9mkTvpS55jlfyn22qJ618itLryxXBhA7Jp3FIlkw=
 v1.3.3 h1:e45DOWDDnd6pZxDPtcB4ALhbnSI1EmL47EtdEic5qo4= -
 v1.3.4 h1:Hhn9fLi13jo58UsyPQMbs8q9WHAU8m+3lYbrhBkvBUk= -
 v1.3.5 h1:W9AlYy3snMQVnE5AIY8tbUdL8iMb9vLwnQOS/IASi4U= -
 v1.3.6 h1:t47x5vThdwgLJzofNsbsAl7gmIiJ7kbDQN5BxwBmwvY= -
github.com/minio/minio-go
 v6.0.5+incompatible h1:qxQQW40lV2vuE9i6yYmt90GSJlT1YrMenWrjM6nZh0Q= h1:7guKYtitv8dktvNUGrhzmNlA5wrAABTQXCoesZdFQO8=
 v6.0.6+incompatible h1:AOPYom8W/kjdsjlsCVYwfb5BELGmkMP7EXhocAm5iME= -
 v6.0.7+incompatible h1:nWABqotkiT/3aLgFnG30doQiwFkDMM9xnGGQnS+Ao6M= -
 v6.0.8+incompatible h1:RBJrzsmxk259C1CvZ9clro73HBPA3zBwFwVePlkom78= -
 v6.0.9+incompatible h1:1GBagCy3VtWteFBwjjNyajSf0JJ/iT0hYVlK8xipsds= -
 v6.0.10+incompatible h1:cAdZRAXBaqI0hU06emlG+6S3wAh52Wr3xRtCK3R7EHc= -
 v6.0.11+incompatible h1:ue0S9ZVNhy88iS+GM4y99k3oSSeKIF+OKEe6HRMWLRw= -
 v6.0.12+incompatible h1:wA5F3AVAzW47K06l3WgXjBPY/Z3QdasUp+cDd6RzUnM= -
 v6.0.13+incompatible h1:SQmjauWGQx5/x2TX47GBeX9xFVEuGB+RJGAVuZzNPtM= -
 v6.0.14+incompatible h1:fnV+GD28LeqdN6vT2XdGKW8Qe/IfjJDswNVuni6km9o= -
github.com/mistifyio/go-zfs
 v1.0.0 h1:F7aMMlUou4K65tJvBjoTNkcL2RdCScCrkuSjMO2Do4U= h1:8AuVvqP/mXw1px98n46wfvcGfQ4ci2FwoAjKYxuo3Z4=
 v2.0.0+incompatible h1:Wz3ho59IuAHiXy6nRto18MIxaff+RgoL89TZmh4+1Gw= -
 v2.1.0+incompatible h1:4QqBc+v+FSw2DloZ8IrPXAx/svvK1Rzcwy/SE1azeS4= -
 v2.1.1+incompatible h1:gAMO1HM9xBRONLHHYnu5iFsOJUiJdNZo6oqSENd4eW8= -
github.com/mitchellh/cli
 v1.0.0 h1:iGBIsUe3+HZ/AD/Vd7DErOt5sU9fa8Uj7A2s1aggv1Y= h1:hNIlj7HEI86fIcpObd7a0FcrxTWetlwJDGcceTlRvqc=
github.com/mitchellh/copystructure
 v1.0.0 h1:Laisrj+bAB6b/yJwB5Bt3ITZhGJdqmxquMKeZ+mmkFQ= h1:SNtv71yrdKgLRyLFxmLdkAbkKEFWgYaq1OVrnRcwhnw=
github.com/mitchellh/go-homedir
 v1.0.0 h1:vKb8ShqSby24Yrqr/yDYkuFz8d0WUjys40rvnGC8aR0= h1:SfyaCUpYCn1Vlf4IUYiD9fPX4A5wJrkLzIz1N1q0pr0=
 v1.1.0 h1:lukF9ziXFxDFPkA1vsr5zpc1XuPDn/wFntq5mG+4E0Y= -
github.com/mitchellh/go-wordwrap
 v1.0.0 h1:6GlHJ/LTGMrIJbwgdqdl2eEH8o+Exx/0m8ir9Gns0u4= h1:ZXFpozHsX6DPmq2I0TCekCxypsnAUbP2oI0UX1GXzOo=
github.com/mitchellh/hashstructure
 v1.0.0 h1:ZkRJX1CyOoTkar7p/mLS5TZU4nJ1Rn/F8u9dGS02Q3Y= h1:QjSHrPWS+BGUVBYkbTZWEnOh3G1DutKwClXU/ABz6AQ=
github.com/mitchellh/iochan
 v1.0.0 h1:C+X3KsSTLFVBr/tK1eYN/vs4rJcvsiLU338UhYPJWeY= h1:JwYml1nuB7xOzsp52dPpHFffvOCDupsG0QubkSMEySY=
github.com/mitchellh/mapstructure
 v1.0.0 h1:vVpGvMXJPqSDh2VYHF7gsfQj8Ncx+Xw5Y1KHeTRY+7I= h1:FVVH3fgwuzCH5S8UJGiWEs2h04kUh9fWfEaFds41c1Y=
 v1.1.0 h1:PoCJ/Ct9du6caE+91v8ov4CLjO4XEBgkPk/dF1v43eo= -
 v1.1.1 h1:0fcGQkeJPHl7DauilpdNG27ZxXHDSg+rbbTpfpniZd8= -
 v1.1.2 h1:fmNYVwqnSfB9mZU6OS2O6GsXM+wcskZDuKQzvN1EDeE= -
github.com/mitchellh/reflectwalk
 v1.0.0 h1:9D+8oIskB4VJBN5SFlmc27fSlIBZaov1Wpk/IfikLNY= h1:mSTlrgnPZtwu0c4WaC2kGObEpuNDbx0jmZXqmk4esnw=
github.com/montanaflynn/stats
 v0.4.0 h1:Ug//pM6DXionEL5D5a8C0PJIXhojyxTFErfBVEEhqDM= h1:wL8QJuTMNUDYhXwkmfOly8iTdp5TEcJFWZD2D7SIkUc=
 v0.5.0 h1:2EkzeTSqBB4V4bJwWrt5gIIrZmpJBcoIRGS2kWLgzmk= -
github.com/mozilla-services/heka
 v0.7.2 h1:Y8ZiWOXYY26GGEw/KOrvdL1XDp6OOazFZfFZX/LpPu4= h1:fBVzlne8jq4zln3gmLzN2ReVpb6guLvIyau8HeC0iVM=
 v0.7.3 h1:WvxQZRIf3CVZuvu9Ci4NoZmudWmXIQ1N1J9P4E8jne4= -
 v0.8.0 h1:tM6kKrD7FeQrAU9WoW4FQInwqCLcuPdOfU7ysXhl/5s= -
 v0.8.1 h1:kbIb5Uo/0klLO2A2S671JXivka6GVXLrSYzKiAjWJkg= -
 v0.8.2 h1:VkU9fz51ZGya1QNRGqHwXZS8qRlK5JAOxkEHjYXM+Yc= -
 v0.8.3 h1:L41mndagpnvuUDCt1+3cBRY7z5zF8KH+5IK6UOb0vXU= -
 v0.9.0 h1:CWHWSWwd4lKhuNv7CY5oysdja59gCdlQyASVWIXnHg8= -
 v0.9.1 h1:tf8CPDjF83xbAAqGevjR/Mro6QUkkqAtbh8TZw91ha8= -
 v0.9.2 h1:tgowoPYv96S/PmtqrZxA3gIuFWCgvdzOnKgFhZJ4gis= -
 v0.10.0 h1:w+y6RPJkU6ZKeNbG1VvK9aSqJm0sru5TYcwOj6ejv8U= -
github.com/mreiferson/go-snappystream
 v0.1.0 h1:FI3WCXQN8tcYKgcord7MsQqRM6kfErt++51rTP6nam0= h1:hPB+SkMcb49n7i7BErAtgT4jFQcaCVp6Vyu7aZ46qQo=
 v0.1.1 h1:pVI3VGpn9432hLMKVKGnS6nKcktgp2ZNnwpwCetJdH0= -
 v0.2.0 h1:qO4LxPBCB3LJeF4AhaY/b+QhO/vhhXg9aMHz24DaebI= -
 v0.2.1 h1:Uj9xUTVpZheXybiuHaXWqrGqF45zM9nHQMg1rSKkcLA= -
 v0.2.2 h1:Rb9OcmL9D939c3r0+EusuD1jEhPpRTh04WxJWwi6fag= -
 v0.2.3 h1:ETSjz9NhUz13J3Aq0NisB/8h0nb2QG8DAcQNEw1T8cw= -
github.com/mssola/user_agent
 v0.1.1 h1:nGq7GHiNaiNu1atYf+KpnxLT65NUoosjNugye0s7zGw= h1:UFiKPVaShrJGW93n4uo8dpPdg1BSVpw2P9bneo0Mtp8=
 v0.1.2 h1:n0RH+xhRoaQUiEx82tBIvgWdFtWEu4pdtkbRSRuLZhw= -
 v0.1.3 h1:6yX2D9nfI/RZdGTLLlqhKOypDM9Kv/G6M6O/kHGuQv4= -
 v0.1.4 h1:rwKJRhpHLGJenKaUVsprqXa0JxDtfaO7UCI0GuMAAy8= -
 v0.2.1 h1:PMEBN38DRFY1tGZr+BqwNegdkTtqSiAr046Tk6R6nKY= -
 v0.4.1 h1:iTUaMpVrb2qWyvUw8UvK3ygWMd2lB1NGuZ1xhpBf1eg= -
 v0.5.0 h1:gRF7/x8cKt8qzAosYGsBNyirta+F8fvYDlJrgXws9AQ= -
github.com/multiformats/go-multiaddr
 v1.2.6 h1:o81W9MkbRtlO4/9ITFx5iLlzgnLEw/iDPD5jiSbNtFw= h1:1JAWc2R8uiQTLrCHI/lmOkXYu5B8025fQbZjq8//YgY=
 v1.2.7 h1:EBphZlqqUCuRjGcaS479YewwDbk+fqyd1E+cg4TllrY= -
 v1.3.0 h1:qv23SBIX9ayNNoGuPFp26xW9cFl3gR4iRXEGFt86aRc= -
 v1.3.1 h1:li9hL2AOm3p+nWbiI5XoOuoQNX+0+nH7pAXNqmXnwZc= -
 v1.3.2 h1:ujjFnBzoBei9JVY8qqxvnSOYlsMPsrNQE+3SSpTK9N8= -
 v1.3.3 h1:6D/tX9e65hCW2mI/jfmre/bVkOQISow/DO3tVtkNL3o= -
 v1.3.4 h1:KeGMWIBvuNcmt8NDxFOQn45mtI262QD6XedFtNxMlyQ= -
 v1.3.5 h1:5R3nGEVlR2Cnd70x4w14uiRfHQHwspZcIN6Yld+oEuM= -
 v1.3.6 h1:U6TYgF6rxYTW/yJMdh3fmSQhL9YBPlSCHOXmnFkCXNw= -
 v1.4.0 h1:xt9fCCmSyTosXSvEhEqYnC75LiaDSdXycwOLJaDGPic= -
github.com/namsral/flag
 v1.7.4-alpha h1:Mo7Jb27IFSrW2WKmU55iw8A9UXagVyv5zcyjAmIFEQ4= h1:OXldTctbM6SWH1K899kPZcf65KxJiD7MsceFUpB5yDo=
 v1.7.4-pre h1:b2ScHhoCUkbsq0d2C15Mv+VU8bl8hAXV8arnWiOHNZs= -
github.com/naoina/toml
 v0.1.0 h1:uDhVQf3Qmc81pKY1PnFOMgAhaAFxZODLxNt4qy1YQkw= h1:NBIhNtsFMo3G2szEBne+bO4gS192HuIYRqfvOWb4i1E=
 v0.1.1 h1:PT/lllxVVN0gzzSqSlHEmP8MJB4MY2U7STGxiouV4X8= -
github.com/nats-io/gnatsd
 v0.9.6 h1:xILwwT4b+o5xV4mJ3ptfrJFQ0WX99vZpwMgjiIAZ20E= h1:nqco77VO78hLCJpIcVfygDP2rPGfsEHkGTUk94uh5DQ=
 v1.0.0 h1:sxHtHzWb9GSziIUToe8IdpA5BJMZfkbGJNKACXfcP6g= -
 v1.0.2 h1:DnLxkTZS2JHlBgt/7KJZQRZO83L7OxrXjKxlHxflEOg= -
 v1.0.4 h1:99kZdSPRfdW75pT3GqVA4b7RXBRFjkMdUOAn8fYHO+4= -
 v1.0.6 h1:E/tfJB32QoEt/IJUNwTqAGMHNKjrKr52b4yvy0gu7dA= -
 v1.1.0 h1:Yo5uA2vay3xMMDysJruJrzogJyv3MBSCYfj+pR2zTD0= -
 v1.2.0 h1:WKLzmB8LyP4CiVJuAoZMxdYBurENVX4piS358tjcBhw= -
 v1.3.0 h1:+5d80klu3QaJgNbdavVBjWJP7cHd11U2CLnRTFM9ICI= -
 v1.4.0 h1:/02WfGM2p1WU8xEapi44bH0Hdh71oEfrHiKiiqetdHM= -
 v1.4.1 h1:RconcfDeWpKCD6QIIwiVFcvForlXpWeJP7i5/lDLy44= -
github.com/nats-io/go-nats
 v1.0.9 h1:B6Z8g23jLAFpbTe3iZhg+errQ/4A61vnrEZFICWgujM= h1:+t7RHT5ApZebkrQdnn6AhQJmhJJiKAvJUio1PiiCtj0=
 v1.1.2 h1:sIFxvIp8KBUZRK25BGBHNTPMelOUTbpAPtb83755ivI= -
 v1.1.6 h1:w42rQ9qyt6IyUXtp9l57KyYIn/1Gl8hCJSNBKg6NAIA= -
 v1.2.0 h1:0YZp58mwYJE1C2WbECmBQjlGzkyjL+UmDYG+Y4efWmE= -
 v1.2.2 h1:CE9FdkrxbV9xqXsnTHW7q8syEdwMaonYX5UsvE0ypKE= -
 v1.3.0 h1:CrvnAwoB2A2Yma+PcM+5tC++3/wswhcy8OvzqbsUXZQ= -
 v1.4.0 h1:HorYtzWLSxkmcEzAFmY6vJ20lFfeAPbrnkoAYgk8GSg= -
 v1.5.0 h1:OrEQSvQQrP+A+9EBBxY86Z4Es6uaUdObZ5UhWHn9b08= -
 v1.6.0 h1:FznPwMfrVwGnSCh7JTXyJDRW0TIkD4Tr+M1LPJt9T70= -
 v1.7.0 h1:oQOfHcLr8hb43QG8yeVyY2jtarIaTjOv41CGdF3tTvQ= -
github.com/nats-io/nats
 v1.0.9 h1:bljCKxqASHpMbrSyWnBsgSA9Aw8AE7OkfAWlLPwGVwA= h1:PpmYZwlgTfBI56QypJLfIMOfLnMRuVs+VL6r8mQ2SoQ=
 v1.1.2 h1:jFFPWU96XpArKDaMbiuf9NiupknTiOlQy26zldaRl7w= -
 v1.1.6 h1:dCWhg0ozPbL1CrXnoIPHjUig8CqccTctbN4wJTpZdMI= -
 v1.2.0 h1:GWpGjidVhfjRJHHaS50xVc8evbjwnfpc/VyROnuMSb8= -
 v1.2.2 h1:silCOicRPZNfnbWpivpCyGWPp3Tpz3kLgN2hNPLDeY0= -
 v1.3.0 h1:SuCyvufVFktz8MEjAO3uykvrV/4k08Kc1NJ2xA7+Op8= -
 v1.4.0 h1:BWGx6P9wCOAljM42Ci9iJMePfLR0/iEo6NZ/jWxatxw= -
 v1.5.0 h1:kyKX8yDb+dYiYuLuRg/3nU7JSfMsDW/nVoimJ9kW5dw= -
 v1.6.0 h1:U5b2apHOTZlUou+NGfCRWG4ZEeivbt2hpsZO4kHKIVU= -
 v1.7.0 h1:1Mw9uuYUx5gs67Rvam3T7Pgn/KiGIrHmVetFAzaUg00= -
github.com/nats-io/nuid
 v1.0.0 h1:44QGdhbiANq8ZCbUkdn6W5bqtg+mHuDE4wOUuxxndFs= h1:19wcPz3Ph3q0Jbyiqsd0kePYG7A95tJPxeL+1OSON2c=
github.com/ncw/swift
 v1.0.35 h1:1Bg5icot4nkYls4Y5Gu4FxX2d8hzlWi8KTYNOFGn0OU= h1:23YIA4yWVnGwv2dQlN4bB7egfYX6YLn0Yo/S6zZO/ZM=
 v1.0.36 h1:U3hGIdY3KHHjXy1TIiSG330H0UiNP/9u58fTTHYj1Iw= -
 v1.0.37 h1:YRISjBl7JO3a3Ojzrr/kJWVFaNW2bj2fy5UwDFZCbFE= -
 v1.0.38 h1:ESdGbt1tXkVhONhnxb78E0dBpykDMalBW69LtAT00/I= -
 v1.0.39 h1:kKWP/n50ohzUiB5m8/Inh0Pi5ftslc1UIyFSmTstSrM= -
 v1.0.40 h1:0c+kzSF82qgP2TvDHwC534eoAMYTRS1jmr6KIMftTk0= -
 v1.0.41 h1:kfoTVQKt1A4n0m1Q3YWku9OoXfpo06biqVfi73yseBs= -
 v1.0.42 h1:ztvRb6hs52IHOcaYt73f9lXYLIeIuWgdooRDhdyllGI= -
 v1.0.43 h1:TZn2l/bPV0CqG+/G5BFh/ROWnyX7dL2D0URaOjNQRsw= -
 v1.0.44 h1:EKvOTvUxElbpDWqxsyVaVGvc2IfuOqQnRmjnR2AGhQ4= -
github.com/nesv/go-dynect
 v0.2.0 h1:6SDjRLQOcZoUdCbkgMANl5LuV772hshhDxLtqAiHLS4= h1:GHRBRKzTwjAMhosHJQq/KrZaFkXIFyJ5zRE7thGXXrs=
 v0.3.0 h1:BIkTLCTNvpYYUFSu83VEScrKIE3hru3DJtYK3M5Li70= -
 v0.3.1 h1:8+yAgl12AZBnJdQjFyAKUreb5Gx46cY0YKKqh1Imri8= -
 v0.4.0 h1:ap1c8ROQzx/egF8tPjJJ7PZrrtbp1gX7dHI45Ds71cw= -
 v0.4.1 h1:SN4bDkQrc75sR15HCr4ugQnu2jX4ZEVrrb2tfzBAcPo= -
 v0.5.0 h1:wXrNmBraO47p8VjEKQp/F4K/PwwOjYSUzBwNlF0tHek= -
 v0.5.1 h1:NcOS6+40wb/ehm5N7LDLuq+bfc6QfC+RQmCK8k87sNA= -
 v0.5.2 h1:X6IJ0frmoDyzhBdyl6vceXJ7Me4PX3ai8/pj6cZhUvE= -
 v0.5.3 h1:ZjwT5DEJG64VZ3Cx+bunLMwJaAM80LjM9wwND1Av2eA= -
 v0.6.0 h1:Ow/DiSm4LAISwnFku/FITSQHnU6pBvhQMsUE5Gu6Oq4= -
github.com/newrelic/go-agent
 v1.8.0 h1:jr/XH0bzGqwf8bxPGdyf+CSi9PRIjMoCsGJDtteLNo0= h1:a8Fv1b/fYhFSReoTU6HDkTYIMZeSVNffmoS726Y0LzQ=
 v1.9.0 h1:DEDLnB+UsGDCcZvMZce3DdyZHL7t400+f2G8eDAZdy0= -
 v1.10.0 h1:bBzikxWb/BvAS013UqiJAvOKN9jqMI0DpCIcqkGjSBM= -
 v1.11.0 h1:jnd8+H6dB+93UTJHFT1wJoij5spKNN/xZ0nkw0kvt7o= -
 v2.0.0+incompatible h1:2LLCGB8HMKZwCAxugdZM1Yy8Neizj4Kq7wS2rFZWoZw= -
 v2.1.0+incompatible h1:fCuxXeM4eeIKPbzffOWW6y2Dj+eYfc3yylgNZACZqkM= -
 v2.2.0+incompatible h1:h7uU1vi+U6Z8TW4az5yewM68QMScsArxezuHvDFgSvU= -
 v2.3.0+incompatible h1:UcQ3vOg6rP5TZbyJ4niqD2Mhwu7HZpNKu+9OzlSHIKU= -
 v2.4.0+incompatible h1:vkvrQxNCtTfZ+KK15zSiOchOVg4B/jrC7bJiXiCk1K4= -
 v2.5.0+incompatible h1:umBF/DtNZEO1ASmD0C8jMaQKD20nZS6os0l/upoLLfo= -
github.com/nicksnyder/go-i18n
 v1.2.0 h1:X6x+iqwFGbtZEtyVJ/s2t3FGO3r4emSUHB/z6Gc3WMk= h1:HrK7VCrbOvQoUAQ7Vpy7i87N7JZZZ7R2xBGjv0j365Q=
 v1.3.0 h1:jAmCijPKtYQy43WmfxnVj4fu2xaaIl/JtW3JmVwfZKQ= -
 v1.4.0 h1:AgLl+Yq7kg5OYlzCgu9cKTZOyI4tD/NgukKqLqC8E+I= -
 v1.5.0 h1:nMTX+o1sp4o7lwnH3FkNswdWEtkzaGRs/DV9NlICNuU= -
 v1.6.0 h1:mNvFF+Tn+2VXDvZcv7Ui/kzwfO6z/2XcJ2ChUHEB3hk= -
 v1.7.0 h1:LomIyLNR8j+q+M0buXFTIqfTCKQ9akx8bmfXnoTN374= -
 v1.8.0 h1:oCz3S6vDjk2an780yCNQc4VhWsPqJrweM0y8V2fzWaw= -
 v1.8.1 h1:omZfCSJYIaw1kTZWSLAUTJ1m6C7HSwhXuYrbDf3Yeo0= -
 v1.9.0 h1:p6IgYlHtynxFjBJsriZWoldnpU/ibnYNwzmYz0jakxU= -
 v1.10.0 h1:5AzlPKvXBH4qBzmZ09Ua9Gipyruv6uApMcrNZdo96+Q= -
github.com/nlopes/slack
 v0.0.1 h1:TyFfv41qapa3DVcEVsvff3wuOcbKaK8T9UU4JMoOa88= h1:jVI4BBK3lSktibKahxBF74txcK2vyvkza1z/+rRnVAM=
 v0.1.0 h1:YnVhdQvWT/m0TDh3VNpSoCBDlD7Y4pz1qUqb/NrNyUs= -
 v0.2.0 h1:ygNVH3HWrOPFbzFoAmRKPcMcmYMmsLf+vPV9DhJdqJI= -
 v0.3.0 h1:jCxvaS8wC4Bb1jnbqZMjCDkOOgy4spvQWcrw/TF0L0E= -
 v0.4.0 h1:OVnHm7lv5gGT5gkcHsZAyw++oHVFihbjWbL3UceUpiA= -
 v0.5.0 h1:NbIae8Kd0NpqaEI3iUrsuS0KbcEDhzhc939jLW5fNm0= -
github.com/nsqio/go-nsq
 v0.3.6 h1:4nqoL5hJtyGADYVxego+TzDsLGBgf+02D0sK/ig7/rg= h1:XP5zaUs3pqf+Q71EqUJs3HYfBIqfK6G83WQMdNN+Ito=
 v0.3.7 h1:FgKbkpO5qmpsgxbHNgAugYm9un4FCNQp/zXoG4qq65Y= -
 v1.0.0 h1:twXxb4R9kkSxUM1IIdpqLm0GT7B6n3GI2vqlxcSv0Cw= -
 v1.0.1 h1:5uQyCSip18qDK4j10TOWEVHkoEHS4srfizDSRhmNSRQ= -
 v1.0.2 h1:CyjU1jbfEarq+1+LDW1NsxHjRp5VtFrr3Em2eIiaP58= -
 v1.0.3 h1:GXn2Qky0n8ynpIkpQcfIcbacjVUpjLayX2YPBUwHPKY= -
 v1.0.4 h1:3e0R+UeSffho3OFQQQGSzCXIbaSb6SRz2j5j16BuIbE= -
 v1.0.5 h1:jsXRL8a5q83ui9GQ5uZeoAx5DU4X3hEf045EJr6oU8E= -
 v1.0.6 h1:h1AmKn7BbrNjUDPJPbWBOQW00ffdWhmvtbbMy1x/FNg= -
 v1.0.7 h1:O0pIZJYTf+x7cZBA0UMY8WxFG79lYTURmWzAAh48ljY= -
github.com/ogier/pflag
 v0.0.1 h1:RW6JSWSu/RkSatfcLtogGfFgpim5p7ARQ10ECk5O750= h1:zkFki7tvTa0tafRvTBIZTvzYyAu6kQhPZFnshFFPE+g=
github.com/oleiade/reflections
 v1.0.0 h1:0ir4pc6v8/PJ0yw5AEtMddfXpWBXg9cnG7SgSoJuCgY= h1:RbATFBbKYkVdqmSFtx13Bb/tVhR0lgOBXunWTZKeL4w=
github.com/olekukonko/tablewriter
 v0.0.1 h1:b3iUnf1v+ppJiOfNX4yxxqfWKMQPZR5yoh8urCTFX88= h1:vsDQFd/mU46D+Z4whnwzcISnGGzXWMclvtLoiIKAKIo=
github.com/olivere/elastic
 v6.2.6+incompatible h1:DABh8qMNCjk+X7O4nbX6A7GA8Aa7K6Z98U8rnUlq81Q= h1:J+q1zQJTgAz9woqsbVRqGeB5G1iqDKVBWLNSYW8yfJ8=
 v6.2.7+incompatible h1:T2g3tZHKopnW1WKgkFCXp71MzPSTFVSgZLO+2gzhzNs= -
 v6.2.8+incompatible h1:DOvOz8+bJZpp63gpeBd9LBo+mqFCoIZlOCE0dcbOC8g= -
 v6.2.10+incompatible h1:O0l73yD32hMuwJ7hYONlQCqBrR2cCZGkWP7dBtdA8Qs= -
 v6.2.11+incompatible h1:XpUEQm7v1YkDgZcG2Bc67oxreOm05T4dD7LWMTclwXo= -
 v6.2.12+incompatible h1:JJ9FxBH/CkfeAXQbyUI8FqzC2vPivNiAXseD2ClQv5Y= -
 v6.2.13+incompatible h1:CtRJjRENblXPfJ1F9T4D+NTvr/oqvbm/U58HXshhp1M= -
 v6.2.14+incompatible h1:k+KadwNP/dkXE0/eu+T6otk1+5fe0tEpPyQJ4XVm5i8= -
 v6.2.15+incompatible h1:j3rfMOkDbo53vnD8mb1Aa89O13RawD/l0W2xSji9FwU= -
 v6.2.16+incompatible h1:+mQIHbkADkOgq9tFqnbyg7uNFVV6swGU07EoK1u0nEQ= -
github.com/onsi/ginkgo
 v1.1.0 h1:9Dna3pyFNjgbQbFqOeXVgGmjymViv1/r/ArPw4+LgO8= h1:lLunBs/Ym6LB5Z9jYTR76FiuTmxDTDusOGeTQH+WWjE=
 v1.1.1 h1:x3juN0zCSyGMynu6Q/HH/sYrK010QanOlQe8cZS6WHU= -
 v1.2.0-beta h1:3Teuu5bbLA4FDE2CLKvLXUlj+jrdNydde2AvoY4ssys= -
 v1.2.0 h1:PpLjPPi/pzx5+cUQ5bMEOa+Dd10mtqsg67lj9lQlqMA= -
 v1.3.0 h1:dIq/ph1T87+3AyCQgKFCIBlBWKqOd/livesOgD0g2wA= -
 v1.3.1 h1:mUZgagGdExHOcJ05DSUfZ0B1EBAzzbWSqpZZuZYuoIo= -
 v1.4.0 h1:n60/4GZK0Sr9O2iuGKq876Aoa0ER2ydgpMOBwzJ8e2c= -
 v1.5.0 h1:uZr+v/TFDdYkdA+j02sPO1kA5owrfjBGCJAogfIyThE= -
 v1.6.0 h1:Ix8l273rp3QzYgXSR+c8d1fTG7UPgYkOSELPhiY/YGw= -
 v1.7.0 h1:WSHQ+IS43OoUrWtD1/bbclrwK8TTH5hzp+umCiuxHgs= -
github.com/onsi/gomega
 v1.1.0 h1:e3YP4dN/HYPpGh29X1ZkcxcEICsOls9huyVCRBaxjq8= h1:C1qb7wdrVGGVU+Z6iS04AVkA3Q65CEZX59MT0QO5uiA=
 v1.2.0 h1:tQjc4uvqBp0z424R9V/S2L18penoUiwZftoY0t48IZ4= -
 v1.3.0 h1:yPHEatyQC4jN3vdfvqJXG7O9vfC6LhaAV1NEdYpP+h0= -
 v1.4.0 h1:p/ZBjQI9G/VwoPrslo/sqS6R5vHU9Od60+axIiP6WuQ= -
 v1.4.1 h1:PZSj/UFNaVp3KxrzHOcS7oyuWA7LoOY/77yCTEFu21U= -
 v1.4.2 h1:3mYCb7aPxS/RU7TI1y4rkEn1oKmPRjNJLNEXgw7MH2I= h1:ex+gbHU/CVuBBDIJjb2X0qEXbFg53c61hWP/1CpauHY=
 v1.4.3 h1:RE1xgDvH7imwFD45h+u2SgIfERHlS2yNG4DObb5BSKU= -
github.com/opencontainers/go-digest
 v1.0.0-rc0 h1:YHPGfp+qlmg7loi376Jk5jNEgjgUUIdXGFsel8aFHnA= h1:cMLVZDEM3+U2I4VmLI6N8jQYUd2OVphdqWwCJHrFt2s=
 v1.0.0-rc1 h1:WzifXhOVOEOuFYOJAW6aQqW0TooG2iki3E3Ii+WN7gQ= -
github.com/opencontainers/image-spec
 v0.5.0 h1:G+GQNhpdujEqYujthmwRpwOZ3qR67gFsxbEEDKGEUrE= h1:BtxoFyWECRxE4U/7sNtV5W15zMzWCbyJoFRP3s7yZA0=
 v1.0.0-rc1 h1:Y7Jr+W0poic7uVRFhKrz8G+UI0KawnlTo/cChNeFrbI= -
 v1.0.0-rc2 h1:JrMI3ko8qAmUZJyWh+gjYMNMB7Hv2PRtZdEM9+QOUTo= -
 v1.0.0-rc3 h1:+2+5sWYeOlm9MRDkl3TnjeFdcvyXXWJDr4JIwoLmDkI= -
 v1.0.0-rc4 h1:bKYlqR9xbFIyaZVEdlWaG/YM5C5Ookv3hnDrkHCa8Cs= -
 v1.0.0-rc5 h1:y+6Q6y07kBaYhJGp5cBj1vEvqrY79H4qG1mYWz2WksU= -
 v1.0.0-rc6 h1:tHxPFsxnoBiEZ4QxNJLNWjqUwfhYZohA+7tGM7UVlU0= -
 v1.0.0-rc7 h1:S9oZRiHHH05dPLkqlz6XmCa08yv9onZ2nhW8QNvowLo= -
 v1.0.0 h1:jcw3cCH887bLKETGYpv8afogdYchbShR0eH6oD9d5PQ= -
 v1.0.1 h1:JMemWkRwHx4Zj+fVxWoMCFm/8sYGGrUVojFA6h/TRcI= -
github.com/opencontainers/runc
 v0.0.8 h1:L2gGxjY2hIGCuOuyP/cGULA5Du+jyOc84qZqIKzejPI= h1:qT5XzbpPznkRYVz/mWwUaVBUv2rmF59PVA73FjuZG0U=
 v0.0.9 h1:F6BWLZMIW7go74TNRTOTnr6OGUVWDot6IuqK+Uvn1EQ= -
 v0.1.0 h1:mR2cOsbXtJDflXhd/x/j3l0wX9tEUkXTnJLzRkREN0g= -
 v0.1.1 h1:GlxAyO6x8rfZYN9Tt0Kti5a/cP41iuiO2yYT0IJGY8Y= -
 v1.0.0-rc1 h1:hpHvA48EjpIWvGtjbZSC2HlRIAUyBN1Yd0ho0tOIFNg= -
 v1.0.0-rc2 h1:IVJPLY0O+EBbCthvT8/aXkNxlTrFIFF+3B0TkjLmOF0= -
 v1.0.0-rc3 h1:tGkPg19g46ZCB9eiKd4Jd0uJ0K17lpsA3ya26UiQFLE= -
 v1.0.0-rc4 h1:kLjrToDU56drfmwQvCTfSafs5zCgUetJEond5bQ0zc8= -
 v1.0.0-rc5 h1:rYjdzMDXVly2Av0RLs3nf/iVkaWh2UrDhuTdTT2KggQ= -
 v1.0.0-rc6 h1:7AoN22rYxxkmsJS48wFaziH/n0OvrZVqL/TglgHKbKQ= -
github.com/opencontainers/runtime-spec
 v0.4.0 h1:Z1hvldJAMZ+FKLtLBn6y52P/lZ6v+XKll5poyC53KQA= h1:jwyrGlmzljRJv/Fgzds9SsS/C5hL+LL3ko9hs6T5lQ0=
 v0.5.0 h1:uu1JuCsXy0gP4iEZ+QpNhBFEfagkcX/TH2QgHdI6lpE= -
 v1.0.0-rc1 h1:C6il5v2hYYjGoItb3OUGHu7Xkmm/1s4VDjobMDLPvvk= -
 v1.0.0-rc2 h1:YVNPxQJsF1eEGlXJC9zaej2RjClDs3xOJsApZR4NheQ= -
 v1.0.0-rc3 h1:sNAphflMIFtEYwISVM+lBUB/a7yaHTFpIh3/SfNWXIY= -
 v1.0.0-rc4 h1:bEUYhLQnVORR+WFacuOZIcjfHCDTpNwve8E6pOLIKGA= -
 v1.0.0-rc5 h1:gUJ82jaA7l+A8tWYQL9Pzr5kh4IbzOP6Qe50sKUzgP0= -
 v1.0.0-rc6 h1:ykBFWOhOawNNCcEes+pUD+QHlU60GTaNgytlxS+OYxo= -
 v1.0.0 h1:O6L965K88AilqnxeYPks/75HLpp4IG+FjeSCI3cVdRg= -
 v1.0.1 h1:wY4pOY8fBdSIvs9+IDHC55thBuEulhzfSgKeC1yFvzQ= -
github.com/opencontainers/selinux
 v1.0.0-rc1 h1:Q70KvmpJSrYzryl/d0tC3vWUiTn23cSdStKodlokEPs= h1:+BLncwf63G4dgOzykXAxcmnFlUaOlkDdmw/CqsW6pjs=
 v1.0.0 h1:AYFJmdZd1xjz5UIb8YpDHthdwAzlM5FVY6PzoNMgAMk= -
github.com/opencontainers/specs
 v0.4.0 h1:xGjEWSfREvjPuK70uK3trtpvB8PsLKaAGbFx1hG3UHA= h1:bfATBL+Vm2NxrM/Y/nmJaR+k3fR3SRg9kgbrzBbCYpI=
 v0.5.0 h1:itG9vYtqBf/NxQi/44EGw7w6FQ+bQZkcKxQ0GaH7Cdo= -
 v1.0.0-rc1 h1:IcoiDMqzj0vxNWZTVYasrBTtaMh879zK8Smx4xRXsXU= -
 v1.0.0-rc2 h1:QKN/tIiRRpth8dZuhnvc+0BaoiPU1goxgZq1n2izr94= -
 v1.0.0-rc3 h1:EuulC3OCfNPMFIwHKrUuV0rTlIFwhOPNkGWVlkj4aS0= -
 v1.0.0-rc4 h1:sw+dZp4BwFXGwwvZiFfFG7eWTM504SMdoh6BxCmFe9I= -
 v1.0.0-rc5 h1:ppqkki/5lcjrcosFMyV7MhHZoR/YqEJTzxrdaM9dCkU= -
 v1.0.0-rc6 h1:lQ5g1z6HbuzP1M/Be3JkPl9qEC0q8+QEahpDHi3d19I= -
 v1.0.0 h1:Bo78vwItQ4jOgGYAzUqjS9Jiwvbc9OVyntawqvc9KDc= -
 v1.0.1 h1:1KLfitICRlMQJ/mT0k5E/gHLvo5yBZd6nS2yLsWHzDI= -
github.com/openshift/origin
 v3.9.0-alpha.2+incompatible h1:RevTsWonVYTP3ITkJo7rDnwiqdc/Z/VJYo9DhFLxzR0= h1:0Rox5r9C8aQn6j1oAOQ0c1uC86mYbUFObzjBRvUKHII=
 v3.9.0-alpha.3+incompatible h1:se/67e+1pLPWTcZvePR7QHdIrZe3w9p8HxIy4NJzwdU= -
 v3.9.0-alpha.4+incompatible h1:J12bNwtGINYEt5aO96k3h2K/ECCCyFhZITMZYAaS0fg= -
 v3.9.0+incompatible h1:vHw2kUaRmyDBU7eglWwnR3dEaWUpthak8ZoaER+BTkg= -
 v3.10.0-alpha.0+incompatible h1:j/cz3HW9LwSH4QTD8wLKOkkrDCqEc3ZhVxA7QCmYMlI= -
 v3.10.0-rc.0+incompatible h1:LhQ3GY32QhsMbGn0VYatZHOrv99zRKsTJCIVl8UrWbg= -
 v3.10.0+incompatible h1:eYdMR5UVG1pMxqrgBdRQcTgDWokb2NVSgroJZWsRsPA= -
 v3.11.0-alpha.0+incompatible h1:8RqgnP1x7+1Ar3NJHNJWX3W6tDsb7V1kt2OYss9qheI= -
 v3.11.0+incompatible h1:DsJmCxDgQzGkW+r+h7rzXSx2HI7HHiIMuYKSOjxVleg= -
 v4.0.0-alpha.0+incompatible h1:MhrqdCD3DadgEz7LGviuIgjWJZ66ggqQJK7dQgYsVq8= -
github.com/openshift/source-to-image
 v1.1.4 h1:AsHVXoUK89of3QO0BdWhP9YGRrPupJmIJymZeKhFuTQ= h1:LR3Zbcy5zfRjqLEftlfYqxirrDYkhrR9VLo+hs76/PI=
 v1.1.5 h1:qiWRlq5Py1MNOe35VwMOECN0xBBfMYCbzKWalGnSH9E= -
 v1.1.6 h1:4+AitFB6lrNP1V9mqoaEGGVsRJXIdmAm8hbeg2BiGJY= -
 v1.1.7 h1:nuCsYqvdKyBVogA21JgWk3NifZ7iTggLeay0RPW6ybI= -
 v1.1.8 h1:R5CkG6b8IOAFM4Ut5kJ/U/VxzV34BfJE9v3aDMBoiUY= -
 v1.1.9 h1:GLHlBULW3SWuEbtLn0jwFAt3iei0i7Re2pMrW3WksTs= -
 v1.1.10 h1:UvEkO45l2JOxQdygB4Jl+hHeLtgjUl6Yry0ogTT63uI= -
 v1.1.11 h1:XrMg3CGmVFwzjUmMTJMTkqZziRKcXCbaL4S6KR9amhk= -
 v1.1.12 h1:mLxuVx5oN4agwNNgYVmw9e9roq4DeIY0oCtMrFibMgc= -
 v1.1.13 h1:kKf1wzAaDrI4Ch2UhBGO0aW8dtDMPIDCZ3ZaTnerxR4= -
github.com/opentracing/basictracer-go
 v1.0.0 h1:YyUAhaEfjoWXclZVJ9sGoNct7j4TVk7lZWlQw5UXuoo= h1:QfBfYuafItcjQuMwinw9GhYKwFXS9KnPs5lxoYwgW74=
github.com/opentracing/opentracing-go
 v0.9.0 h1:4DQ5HRAp5evcmDqWInLpZeANKBXIKQxHTXxclfMwFek= h1:UkNAQd3GIcIGf0SeVgPpRdFStlNbqXla1AfSYxPUl2o=
 v0.10.0 h1:XN6nZRjbLE+GkEuO3rti6ZcnDqpaIF0yKJ1z6rLA63k= -
 v1.0.0 h1:fDGaqxLymyQ1cyyVYBrR6p4MPwKZA7clqxUleobb2VA= -
 v1.0.1 h1:IYN/cK5AaULfeMAlgFZSIBLSpsZ5MRHDy1fKBEqqJfQ= -
 v1.0.2 h1:3jA2P6O1F9UOrWVpwrIo17pu01KWvNWg4X946/Y5Zwg= -
github.com/openzipkin/zipkin-go-opentracing
 v0.2.1 h1:pAArtgscP0jFG9lf4gv0JBqGqOSCXmGJCyxQGBkgnhI= h1:js2AbwmHW0YD9DwIw2JhQWmbfFi/UnWyYwdVhqbCDOE=
 v0.2.2 h1:2PLiliCVlagg3bi13LIEfABAoNCJ9vrXxY2HLbPKztk= -
 v0.2.3 h1:6hxV9HayqBXV6FXYqPDdZQsJFA376pNuZfsOfEX7c0s= -
 v0.2.4 h1:yqgP6E5SqRqfpTikIQ48T1qOptmrp6f9HydXRbwjdwE= -
 v0.3.0 h1:Y0pgC50ZAww7r/q1PSnlXrT68DR4hjsign+4Ixz+Tbg= -
 v0.3.1 h1:QjUuBG9au4XDpsf4BFhWJJoPEARHLl0/cLpqktWhAHE= -
 v0.3.2 h1:29bHr4whILL23QNcqpW5AvhI42bXDGy5a2tPDOa7QqM= -
 v0.3.3 h1:EbvYHyxNeJCm2LonvlIKb9Pg3LxsVAbpZI3iEFTr078= -
 v0.3.4 h1:x/pBv/5VJNWkcHF1G9xqhug8Iw7X1y1zOMzDmyuvP2g= -
 v0.3.5 h1:nZPvd2EmRKP+NzFdSuxZF/FG4Y4W2gn6ugXliTAu9o0= -
github.com/ory-am/common
 v0.0.1 h1:GwAs3QSTunR9c9nQ5s4i6LhrV9SoUDULmnRxnTDm+48= h1:oCYGuwwM8FyYMKqh9vrhBaeUoyz/edx0bgJN6uS6/+k=
 v0.1.0 h1:A9oPp2PRPRpWRVCmqUPaPYMXb24U/nX9plxpLoiz+0E= -
 v0.2.0 h1:Wzb6AXy+RFk5w2pv0mFrXVVETLRI71QgKD2YVEpk7uI= -
 v0.2.1 h1:MAffe/ebupYZ1MswaqdGuZqWTXYjwm8ioS3XzkLHsVU= -
 v0.2.2 h1:qynTwak3lattD3RI89uPqdrzCs1hJInaqiMrqDj/VBg= -
 v0.3.0 h1:CiAAZJ2oCa0Sp1D+TOAO7xw++GhOAt8hpoLMKDBE7lg= -
 v0.4.0 h1:edGPoxYX4hno0IJHXh9TCMUPR6ZcJp+y6aClFYxeuUE= -
github.com/oschwald/geoip2-golang
 v0.1.0 h1:KD3+PHmbJn5yoK5IK9U1o0a9BMeRezQLUKXlGIYUuzs= h1:0LTTzix/Ao1uMvOhAV4iLU0Lz7eCrP94qZWBTDKf0iE=
 v1.0.0 h1:zCPPwfexroPmnABwxzJfHBkdl8bF08Cym4GO90hJpCw= -
 v1.1.0 h1:ACVPz5YqH4/jZkQdsp/PZc9shQVZmreCzAVNss5y3bo= -
 v1.2.0 h1:uPmjb3XMGAHZu7oEX44sf9foBGvmLy4R99OQUjXzStE= -
 v1.2.1 h1:3iz+jmeJc6fuCyWeKgtXSXu7+zvkxJbHFXkMT5FVebU= -
github.com/oschwald/maxminddb-golang
 v0.1.0 h1:NcvxeapDUYTtHv1zehpp+9G3YDYuWMX4Q4Dns8Qbwpg= h1:3jhIUymTJ5VREKyIhWm66LJiQt04F0UCDdodShpjWsY=
 v0.2.0 h1:cdvE3VUWCRdu+tYIBtwbcPWj1A83jZc5CSdGSuhnqO8= -
 v1.0.0 h1:J2B9uC7QU/ySxtQi8UkPyry1JzNGfPyY6YjIHmgTLno= -
 v1.1.0 h1:WoRFEb51Ahbd6PfUNtIjIokFoYqKQJYdWLHRTSiw2D8= -
 v1.2.0 h1:GMRJq8+Qmb4fuybkd3aMJhBx07Bqu0fE3y3pcKcItKw= -
 v1.2.1 h1:1wUyw1BYyCY7E0bbG8lD7P5aPDFIsRr611otw6LOJtM= -
 v1.3.0 h1:oTh8IBSj10S5JNlUDg5WjJ1QdBMdeaZIkPEVfESSWgE= -
github.com/packethost/packngo
 v0.1.0 h1:G/5zumXb2fbPm5MAM3y8MmugE66Ehpio5qx0IhdhTPc= h1:otzZQXgoO96RTzDB/Hycg0qZcXZsWJGJRSXbmEIJ+4M=
github.com/parnurzeal/gorequest
 v0.2.6 h1:6O0X3Tis+5b6XSYHB3shqD2+MEvzPIrjtQQ5xagww5g= h1:3Kh2QUMJoqw3icWAecsyzkpY7UzRfDhbRdTjtNwNiUE=
 v0.2.7 h1:GUjRVflL738No32Qu+OvRNdKyxGhpe2ExRb5DvSxJY8= -
 v0.2.8 h1:5tqfNpdewuJroyhasrtBu11Y86fiu4HmDTII4Ecllz8= -
 v0.2.9 h1:abFRy8OmE8Rs9MOZEgx0MWV4rQkyszgFSF+/vNGo4no= -
 v0.2.10 h1:OOT1XNRaJ/DfE0zqwvMIhBxCAx/QyHBk+8956yKtD7c= -
 v0.2.11 h1:7weAk3u0O/4VdIyVyOVjXQxYyr4M7XhDqIjMTQrcQrk= -
 v0.2.12 h1:ub6AicO4LWdlKjoJG45NmX0W/5wSfj+cXlBobrXE/Sw= -
 v0.2.13 h1:p5XL6G+4eOh5ifLB1SBPTqFyTiqHmDpq3gDquVULa2k= -
 v0.2.14 h1:9VG0N3uWBo1xxs4Fk/EHzGhrRp4UFnksDcX/DYpyZs8= -
 v0.2.15 h1:oPjDCsF5IkD4gUk6vIgsxYNaSgvAnIh1EJeROn3HdJU= -
github.com/patrickmn/go-cache
 v1.0.0 h1:3gD5McaYs9CxjyK5AXGcq8gdeCARtd/9gJDUvVeaZ0Y= h1:3Qf8kWWT7OJRJbdiICTKqZju1ZixQ/KpMGzzAfe6+WQ=
 v2.0.0+incompatible h1:1G02Ver4lZNbrWBHtot9O0Z2Piky5+/ilrJaTIwL/w0= -
 v2.1.0+incompatible h1:HRMgzkcYKYpi3C8ajMPV8OFXaaRUnok+kx1WdO15EQc= -
github.com/paulbellamy/ratecounter
 v0.1.0 h1:FLljiU2IX0ugjoNr1V85RkPMH/od1Rnj/9vnzmTZTfs= h1:Hfx1hDpSGoqxkVVpBi/IlYD7kChlfo5C6hzIHwPqfFE=
 v0.2.0 h1:2L/RhJq+HA8gBQImDXtLPrDXK5qAj6ozWVK/zFXVJGs= -
github.com/pborman/uuid
 v1.2.0 h1:J7Q5mO4ysT1dv8hyrUGHb9+ooztCXu1D8MY8DZYsu3g= h1:X/NO0urCmaxf9VXbdlT7C2Yzkj2IKimNn4k+gtPdI/k=
github.com/pebbe/zmq4
 v1.0.0 h1:D+MSmPpqkL5PSSmnh8g51ogirUCyemThuZzLW7Nrt78= h1:7N4y5R18zBiu3l0vajMUWQgZyjv464prE8RCyBcmnZM=
github.com/pelletier/go-toml
 v0.3.2 h1:cVBP3JbBPviw5hB/hOXp7YqtoOndpAEOb7CiCVof/NY= h1:5z9KED0ma1S8pY6P1sdut58dfprrGBbd/94hg7ilaic=
 v0.3.3 h1:H3CcHWqJhLImSTzDMfBfNyEV4MNxd4lVkeHX2I9iXIA= -
 v0.3.4 h1:odJeysv0AIeTOBFqnN3Ad+ahpfT/Sj557a4E9jLPG7g= -
 v0.3.5 h1:vtEp5CVqyTzO3Qvi3fU5uv+rm9pdq+2YpdWAPrGL26A= -
 v0.4.0 h1:vQTrQ6pGYOySEPMpdOyUcWbmZfqDuoyYzVjy1zNthPg= -
 v0.5.0 h1:4JciwWR3v6XGcQibYe+58htiJN9TzM6P18uLetto3gw= -
 v1.0.0 h1:QFDlmAXZrfPXEF6c9+15fMqhQIS3O0pxszhnk936vg4= -
 v1.0.1 h1:0nx4vKBl23+hEaCOV1mFhKS9vhhBtFYWC7rQY0vJAyE= -
 v1.1.0 h1:cmiOvKzEunMsAxyhXSzpL5Q1CRKpVv0KQsnAIcSEVYM= -
 v1.2.0 h1:T5zMGML61Wp+FlcbWjRDT7yAxhJNAiPPLOFECq181zc= -
github.com/peterbourgon/diskv
 v1.0.0 h1:bRU92KzrX3TQ6IYobfie/PnZkFC+1opBfHpf/PHPDoo= h1:uqqh8zWWbv1HBMNONnaR/tNboyR3/BZd58JJSHlUSCU=
 v2.0.0+incompatible h1:WM3au3ZA2yuY/ByeDAbOPEsl0HTVfB3v5oBUOyKWnlk= -
 v2.0.1+incompatible h1:UBdAOUP5p4RWqPBg048CAvpKN+vxiaj6gdUUzhl4XmI= -
github.com/peterh/liner
 v1.0.0 h1:HmYbuOWNntSzDEbCLX29yAdt/jAaG2V6lTeDXbniOEs= h1:xIteQHvHuaLYG9IFj6mSxM0fCKrs34IrEQUhOYuGPHc=
 v1.1.0 h1:f+aAedNJA6uk7+6rXsYBnhdo4Xux7ESLe+kcuVUF5os= h1:CRroGNssyjTd/qIG2FyxByd2S8JEAZXBl4qUrZf8GS0=
github.com/pierrec/lz4
 v1.0.1 h1:w6GMGWSsCI04fTM8wQRdnW74MuJISakuUU0onU0TYB4= h1:pdkljMzZIN41W+lC3N2tnIh5sFi+IEE17M5jbnwPHcY=
 v2.0.1+incompatible h1:goKmVq5xV/0uAnq7ldSRSHSkc3yIZaSNQry66C/sE/4= -
 v2.0.2+incompatible h1:6spEXYEkGG74KeVRPzvSU0Fa3xO9DGO0bJcA6uIfwo8= -
 v2.0.3+incompatible h1:h0ipQUMRrnr+/HHhxhceftyXk4QcZsmxSNliSG75Bi0= -
 v2.0.4+incompatible h1:ugy0aRlLvrxupbCD6xnVilNrEOM89+ftxz+4FsUX0BU= -
 v2.0.5+incompatible h1:2xWsjqPFWcplujydGg4WmhC/6fZqK42wMM8aXeqhl0I= -
github.com/pilu/traffic
 v0.2.0 h1:guerUOB3OtcJb0pEJ/RK2BCrdduWUtpV362rdjyfBDA= h1:pPbakW2fofyQaXrh319Xb9TJpM8gjs/vp+71CztjeK0=
 v0.3.0 h1:e57u2e42EqcmXqoDtBZNgS3VY4qVdcWgA4c5qg/dHmI= -
 v0.4.0 h1:OAyV2JDz0k99twFfjlmLuFlPiwk394BCx0hMc8j8Wm0= -
 v0.5.0 h1:hm7c02s5gzbHdmZirbDCSJPeZYsm9DMejdyaztJfxLs= -
 v0.5.1 h1:aKL6eyVR+fI7ERMgMcujX2DeUYgXPKjNVk7HO8/8xAk= -
 v0.5.2 h1:T5szRLz2IeBgTa94TJnYzIPpzeja0O/wOEbeG/m9ODE= -
 v0.5.3 h1:gzDC+/uF1JQNnpcjUhVnw3z9xTBKx9y9FLtdLkHrIbQ= -
github.com/pkg/errors
 v0.2.0 h1:eqJNyK4um6+PDsjJ/fQtIa5T6A2TPq4g2rV/YfKxiNQ= h1:bwawxfHBFNV+L2hUp1rHADufV3IMtnDRdf1r5NINEl0=
 v0.3.0 h1:vGTT2whb6g8by61YzK2lWzUOI5Ii8A/J6BpAF8EL35g= -
 v0.4.0 h1:WGkfvLuWDIBfhb2zl49//ybqmEN9P3OcOrPYUGJrPIo= -
 v0.5.0 h1:kOgBq3ZvaGheGMUHLjQpZwRq5jNrauDRNA735/cpzo8= -
 v0.5.1 h1:IHiy+E7QjmNYS4wzsr+PbDPot3/5g3LddPReiKtiOKM= -
 v0.6.0 h1:O89Tl73EiJ8Lvu4nVRs7uW10mpZnajMT+y5a0NIjLgA= -
 v0.7.0 h1:WLW8U1O88/efNaH7+8C+KBNkZRGD+WCmNlC0b06x/Ig= -
 v0.7.1 h1:0XSZhzhcAUrs2vsv1y5jaxWejlCCgvxI/kBpbRFMZ+o= -
 v0.8.0 h1:WdK/asTD0HN+q6hsWO3/vpuAkAr+tw6aNJNDFFf0+qw= -
 v0.8.1 h1:iURUrRGxPUNPdy5/HRSm+Yj6okJ6UtLINN0Q9M4+h3I= -
github.com/pkg/profile
 v1.0.0 h1:o++U+8vy+VZt0fK9G2vjGmvh+Zknftwbl/TyP787onA= h1:hJw3o1OdXxsrSjjVksARp5W95eeEaEfptyVZyv6JUPA=
 v1.1.0 h1:TbomchXmJaOHG1UPkn11R6MStSYlI0AH/HRoRS0Wmpw= -
 v1.2.0 h1:rtDCCNu0JSILFwbaxmdUzkrwUNetI++CY5pEIhv6WB4= -
 v1.2.1 h1:F++O52m40owAmADcojzM+9gyjmMOY/T4oYJkgFDH8RE= -
github.com/pkg/sftp
 v1.8.0 h1:SJ2EX5aeifvl4zzbci3urQbr5p7Xc/A7/Ia9T8ahhNM= h1:NxmoDg/QLVWluQDUYG7XBZTLUpKeFa8e3aMf1BfjyHk=
 v1.8.1 h1:lBNmfMtetdLwsQv4Bh+DAWADK2yYljdc0qmjMQUb4qc= -
 v1.8.2 h1:3upwlsK5/USEeM5gzIe9eWdzU4sV+kG3gKKg3RLBuWE= -
 v1.8.3 h1:9jSe2SxTM8/3bXZjtqnkgTBW+lA8db0knZJyns7gpBA= -
 v1.9.0 h1:ljQo4ePyIKAOXeUeoFXiZQV81QsI+dqErOJ7369VC4w= -
 v1.9.1 h1:dfh6lpPQzEP8ZRaRZww2DL074+Ov+DwaFy7bdncZpiM= -
 v1.10.0 h1:DGA1KlA9esU6WcicH+P8PxFZOl15O6GYtab1cIJdOlE= -
github.com/pmezard/go-difflib
 v1.0.0 h1:4DBwDE0NGyQoBHbLQYPwSUPoCMWR5BEzIk/f1lZbAQM= h1:iKH77koFhYxTK1pcRnkKkqfTogsbg7gZNVY4sRDYZ/4=
github.com/pmylund/go-cache
 v1.0.0 h1:jbJMNhn4LhBfb3dRejPlnjxSiokDt4qO5NWt8mMi+UE= h1:hmz95dGvINpbRZGsqPcd7B5xXY5+EKb5PpGhQY3NTHk=
 v2.0.0+incompatible h1:5ZNJ3XnuQrCjvRXItcjhbD27dVFEk7lVqJHbuI696qc= -
 v2.1.0+incompatible h1:n+7K51jLz6a3sCvff3BppuCAkixuDHuJ/C57Vw/XjTE= -
github.com/pquerna/otp
 v1.0.0 h1:TBZrpfnzVbgmpYhiYBK+bJ4Ig0+ye+GGNMe2pTrvxCo= h1:Zad1CMQfSQZI5KLpahDiSUX4tMMREnXw98IvL1nhgMk=
 v1.1.0 h1:q2gMsMuMl3JzneUaAX1MRGxLvOG6bzXV51hivBaStf0= -
github.com/pressly/chi
 v3.2.0+incompatible h1:XbMUwNtES9qOkJtA5ynrxmOjAAFEEvxrZCTnEojlxiw= h1:s/kslmeFE633XtTPvfX2olbs4ymzIHxGGXmEJ/AvPT8=
 v3.2.1+incompatible h1:SZM6/GYakwRbHrP9dnvKNOlxQ9vRA63ARZcP0FNgbek= -
 v3.3.0+incompatible h1:9ptvP/vLACB8IJkNfwzwQwyK38MONUVoK9Bzy6YtxQ8= -
 v3.3.1+incompatible h1:0flC4nrZV9ojesPGYq31Sg5dZSDBp/P7PfcpfnBy3KA= -
 v3.3.2+incompatible h1:OR1IvWoy2hmGsEw7OFB9nt0f4eet9vimbWIp0qL1dSA= -
 v3.3.3+incompatible h1:fc66b0mPg4Dx5Pr86WSsXv0x37dSX6pH0p38GZsvCtU= -
 v3.3.4+incompatible h1:l6iaLk5QzcVz0kwZM9Fvg9W+sJdIvjukKPZQOmn/ywI= -
 v4.0.0-rc2+incompatible h1:VlXJulAN3RV329W+zhMJ1ydUeaSrpjeUYa2Mg4GOSEY= -
 v4.0.0+incompatible h1:yoa5lMe/USUYUY+EW0YNyQwEB/SKm8FRSLBsMJ4DKJA= -
 v4.0.1+incompatible h1:goCMOEqf5UxDSWHd2iZ/R4PvYixQ+S+LXT4yz6ZKi/s= -
github.com/prometheus/client_golang
 v0.8.0 h1:1921Yw9Gc3iSc4VQh3PIoOqgPCZS7G/4xQNVUp8Mda8= h1:7SWBe2y4D6OKWSNQJUaRYU/AaXPKyh/dDVn+NZz0KFw=
 v0.9.0-pre1 h1:AWTOhsOI9qxeirTuA0A4By/1Es1+y9EcCGY6bBZ2fhM= -
 v0.9.0 h1:tXuTFVHC03mW0D+Ua1Q2d1EAVqLTuggX50V0VLICCzY= -
 v0.9.1 h1:K47Rk0v/fkEfwfQet2KWhscE0cJzjgCCDBG2KHZoVno= -
 v0.9.2 h1:awm861/B8OKDd2I/6o1dy3ra4BamzKhYOiGItCeZ740= h1:OsXs2jCmiKlQ1lTBmv21f2mNfw4xf/QclQDMrYNZzcM=
github.com/prometheus/common
 v0.1.0 h1:IxU7wGikQPAcoOd3/f4Ol7+vIKS1Sgu08tzjktR4nJE= h1:TNfzLD0ON7rHzMJeJkieUDPYmFC7Snx/y86RQel1bk4=
 v0.2.0 h1:kUZDBDTdBVBYBj5Tmh2NZLlF60mfjA27rM34b+cVwNU= -
github.com/prometheus/prometheus
 v2.3.2+incompatible h1:EekL1S9WPoPtJL2NZvL+xo38iMpraOnyEHOiyZygMDY= h1:oAIUtOny2rjMX0OWN5vPR5/q/twIROJvdqnQKDdil/s=
 v2.4.0-rc.0+incompatible h1:y2TC7cQXVpmBhDAsux3vaIQeEHQyojzxkOiPeyEPPs8= -
 v2.4.0+incompatible h1:GFdMJQKpo7lM4pvnzJAeXZeGdwTlh4PNO6VYFd+XKHY= -
 v2.4.1+incompatible h1:d/sYPIIWAzSzPfIQk/lrQKNFGrO4dGUO1iO3rCqUWU8= -
 v2.4.2+incompatible h1:IpbpeZAXsg39pqRThfPHoNRYjIyInnUFS26rPVkUXYk= -
 v2.4.3+incompatible h1:hQEvPnUF8oiGkiQMllJiOkzSyoQev1v2nQZaf6/z/4g= -
 v2.5.0-rc.0+incompatible h1:c2H9Meo2yAv+u13Vbin+qjnBX5Bd1F+tQ4eqDS71GUk= -
 v2.5.0-rc.1+incompatible h1:z3LQlpePfz4R3obamJZpK0EBDHj69kx69W5fS+PIScM= -
 v2.5.0-rc.2+incompatible h1:5VQ4aEgo94P1NYFgcSGBLYJp55weNrkWukrP22Xne/U= -
 v2.5.0+incompatible h1:7QPitgO2kOFG8ecuRn9O/4L9+10He72rVRJvMXrE9Hg= -
github.com/rackspace/gophercloud
 v0.0.0 h1:9gsnUzbPadIuEDlHeD+0bPceCUeef4CgD2uOJIDYqoU= h1:4bJ1FwuaBZ6dt1VcDX5/O662mwR8GWqS4l68H6hkoYQ=
 v0.1.0 h1:kNU6XTEI+99Qt8LEgcn1LQqw6SVK2dy0rQ8LuXagD2Y= -
 v1.0.0 h1:dI8jqEOOanEOgOSYcfvBRnETdeScNtlWg3KlJ4guUVQ= -
github.com/rakyll/statik
 v0.1.0 h1:KpbIQtEGvwoA9scoyXPXa3ywkGd8XdpA3OuUcgMVVRk= h1:OEi9wJV/fMUAGx1eNjq75DKDsJVuEv1U0oYdX6GX8Zs=
 v0.1.1 h1:fCLHsIMajHqD5RKigbFXpvX3dN7c80Pm12+NCrI3kvg= -
 v0.1.2 h1:ckeqyWXP9hT/2u8Xfi4L02B2RFlqr6fdedIPgohOAHM= -
 v0.1.3 h1:H/5HK3yNM7sDzOiMQtC2Q1N69hl+KxzomBBWus662LU= -
 v0.1.4 h1:zCS/YQCxfo/fQjCtGVGIyWGFnRbQ18Y55mhS3XPE+Oo= -
 v0.1.5 h1:Ly2UjURzxnsSYS0zI50fZ+srA+Fu7EbpV5hglvJvJG0= -
github.com/rancher/go-rancher
 v0.1.0 h1:YIKWwe5giu2WICfyCcGqX+m4XTRbMpA8vzLxl1Kwb7w= h1:7oQvGNiJsGvrUgB+7AH8bmdzuR0uhULfwKb43Ht0hUk=
github.com/revel/revel
 v0.13.1 h1:kZ3SI4QSR02D4WPQ1NhiIY12L7ltmjcpNj1q+u4SGvM= h1:VZWJnHjpDEtuGUuZJ2NO42XryitrtwsdVaJxfDeo5yc=
 v0.14.0 h1:uLTHZ9pnJ7Xc0eR7BsG7LojTei2HYUPn4fEkX2PDiXk= -
 v0.15.0 h1:ajBOsYPmfgcNYwbYAw8J/AlqZ4TxMxjOh7aMiPM7eq4= -
 v0.16.0 h1:8+VeZkYB5VU04yZHh0QR3gByFv5Qy1ro3zuUPmvhlDc= -
 v0.17.1 h1:4SMsZvP71mw/7XiDzMxPnTCodXPBtS+BZe+9j1VpKZw= -
 v0.18.0 h1:KplMZj+V5ENVbXMkR3dSEuv4KSuyWx6DcJHgWX9C7lE= -
 v0.19.0 h1:lnmpTIHvMudr+d4wPpT2pM6Pryicloh7fzPZl2hB5Tc= -
 v0.19.1 h1:q9AaxiuiCA87fjFSlb0UGYqRaIOdasn0EyCXvbQw+F0= -
 v0.20.0 h1:P6t74V9Mar6k9C4lW1M280hZBW4sLCNs9r/taKZD4wc= -
 v0.21.0 h1:E6kDJmpJSDb0F8XwbyG5h4ayzpZ+8Wcw2IiPZW/2qSc= -
github.com/rjeczalik/notify
 v0.9.0 h1:xJX3IQ09+O0qLAv4YdYe03EwYRyM7NPuC5O7Mc6/Jv4= h1:rKwnCoCGeuQnwBtTSPL9Dad03Vh2n40ePRrjvIXnJho=
 v0.9.1 h1:CLCKso/QK1snAlnhNR/CNvNiFU2saUtjV0bx3EwNeCE= -
 v0.9.2 h1:MiTWrPj55mNDHEiIX5YUSKefw/+lCQVoAFmD6oQm5w8= h1:aErll2f0sUX9PXZnVNyeiObbmTlk5jnMoCa4QEjJeqM=
github.com/rlmcpherson/s3gof3r
 v0.4.0 h1:uJxvaFN1H3MAVH1EAPR9xG4lik4JqczJO+KLEd87YCw= h1:s7vv7SMDPInkitQMuZzH615G7yWHdrU2r/Go7Bo71Rs=
 v0.4.1 h1:qK/Lu3Jc+0EqDCDZjMV3PW480LbKBYlbW1xV1cnLPws= -
 v0.4.2 h1:XqZhnGh6yE1ejCfildCo7LlmmxYYHuhI4RA+ib52AII= -
 v0.4.3 h1:DiT5JvRyTijNCoRQ+SwHSz5xQ/ROUEsGg7cHL/Ch4/I= -
 v0.4.6 h1:KajchGdkn6HkJ9Ju+yWxFobnxHAGXZ5CWj9liBqs4CI= -
 v0.4.7 h1:UVFyVf9Is6a07FN4O4OCj4M1w0QKynVAo0YkSmWlqiQ= -
 v0.4.8 h1:3OWuvS9ZNS90ZVk1QhLXqR5M9eGwh7R4llpUiMFvZQM= -
 v0.4.9 h1:WXDLlYDUjZih/f5EyVfjSK5vousoE/a/b0d/hjNkqPU= -
 v0.4.10 h1:yISvRaERms9AbHNh7p7TIRv3UcsURoNANiuGAdj1kQo= -
 v0.5.0 h1:1izOJpTiohSibfOHuNyEA/yQnAirh05enzEdmhez43k= -
github.com/rogpeppe/fastuuid
 v1.0.0 h1:f5eq2L8Y87sP63CaPojeD05ON4/AEe/wejW/jp8N6QQ= h1:jVj6XXZzXRy/MSR5jhDC/2q6DgLz+nrA6LYCDYWNEvQ=
github.com/rs/cors
 v1.3.0 h1:R0sy4XekGcOFoby9D76NXXg2birJ3WFkzGvXF9Kn3xE= h1:gFx+x8UowdsKA9AchylcLynDq+nNFfI8FkUZdN/jGCU=
 v1.4.0 h1:98SZukVonBOdXatRLa6GSAtp+IeOjY+nmdEZAxImXXc= -
 v1.5.0 h1:dgSHE6+ia18arGOTIYQKKGWLvEbGvmbNE6NfxhoNHUY= -
 v1.6.0 h1:G9tHG9lebljV9mfp9SNPDL36nCDxmo3zTlAf1YgvzmI= -
github.com/rs/xid
 v1.1.0 h1:9Z322kTPrDR5GpxTH+1yl7As6tEHIH9aGsRccl20ELk= h1:+uKXf+4Djp6Md1KODXJxgGQPKngRmWyn10oCKFzNHOQ=
 v1.2.0 h1:qRPemPiF/Pl06j+Pp5kjRpgRmUJCsfdPcFo/LZlsobA= -
 v1.2.1 h1:mhH9Nq+C1fY2l1XIpgxIiUOfNpRBYH1kKcr+qfKgjRc= -
github.com/rs/zerolog
 v1.7.0 h1:mVEg9/3WVlGTfXcwDp7iuspAsvVzq5k15RiGAZbBgwQ= h1:YbFCdg8HfsridGWAh22vktObvhZbQsZXe4/zB0OKkWU=
 v1.7.1 h1:zWBba9nquCvfHc7shNsnucF/E5ZzDnK5h5PPiuh9Waw= -
 v1.8.0 h1:Oglcb4i6h42uWacEjomB2MI8gfkwCwTMFaDY3+Vgj5k= -
 v1.9.0 h1:h+fPIJoX2FeL8y0m9EZdm5UN/Zn9uxl/gaNKBlco9qg= -
 v1.9.1 h1:AjV/SFRF0+gEa6rSjkh0Eji/DnkrJKVpPho6SW5g4mU= -
 v1.10.0 h1:roFDW4AgYGbHnTOAMZ2K8mHJZ/7bSj7txPfvbABIj88= -
 v1.10.1 h1:/oNUEYN/Fmd9vIlqptUkYgz2yB1oL8x4AExTjN6/wj8= -
 v1.10.2 h1:t4oASYf49zTOqUIx+nfDaC0pRnLeupbWTYfGy0CCPpg= -
 v1.10.3 h1:Gbm4pmo3YF7QxRwoNAKvf33oB/bGMIVunAVNJRxQvdg= -
 v1.11.0 h1:DRuq/S+4k52uJzBQciUcofXx45GrMC6yrEbb/CoK6+M= -
github.com/russross/blackfriday
 v1.5.1 h1:B8ZN6pD4PVofmlDCDUdELeYrbsVIDM/bpjW3v3zgcRc= h1:JO/DiYxRf+HjHt06OyowR9PTA263kcR/rfWxYHBV53g=
 v1.5.2 h1:HyvC0ARfnZBqnXwABFeSZHpKvJHJJfPz81GNueLj0oo= -
 v2.0.0+incompatible h1:cBXrhZNUf9C+La9/YpS+UHpUT8YD6Td9ZMSU9APFcsk= -
github.com/ryanuber/columnize
 v0.1.0 h1:AhdWuWqFv/eKkWx5Z9c34o52PhVO2OmzsGdA64GCDQY= h1:sm1tb6uqfes/u+d4ooFouqFdy9/2g9QGwK3SQygK0Ts=
 v1.1.0 h1:9t+qOsBdd6Vx3XbIXrusfA/k0cp2yDw5kJ/PGJOmAJY= -
 v1.1.1 h1:kaLR0w/IEQSUuivlqIGTq3RXnF7Xi5PfA2ekiHVsvQc= -
 v2.0.0+incompatible h1:2FyQ5ZqsIAhAkEzkLk1ucHyBHuIfquuJMX3XHviwtv0= -
 v2.0.1+incompatible h1:sbS5KRxW9y3/yehqmCOkFyrtZi3Aa3CMLPHvv1RJUso= -
 v2.1.0+incompatible h1:j1Wcmh8OrK4Q7GXY+V7SVSY8nUWQxHW5TkBe7YUl+2s= -
github.com/ryanuber/go-glob
 v1.0.0 h1:iQh3xXAumdQ+4Ufa5b25cRpC5TYKlno6hsv6Cb3pkBk= h1:807d1WSdnB0XRJzKNil9Om6lcp/3a0v4qIHxIXzX/Yc=
github.com/satori/go.uuid
 v1.0.0 h1:6QDKTa2a+CpXmqIFypEOKZUreVG3iCcrb8vbCkHTDsY= h1:dA0hQrYB0VpLJoorglMZABFdXlWrHn1NEOzdhQKdks0=
 v1.1.0 h1:B9KXyj+GzIpJbV7gmr873NsY6zpbxNy24CBtGrk7jHo= -
 v1.2.0 h1:0uYX9dsZ2yD7q2RtLRtPSdGDWzjeM3TbMJP9utgA0ww= -
github.com/sendgrid/sendgrid-go
 v3.0.6+incompatible h1:UPOeYa3p9xTiRf1DL5Xwxdxh/VAbD2SP9VdXiiAsTpo= h1:QRQt+LX/NmgVEvmdRw0VT/QgUn499+iza2FnDca9fg8=
 v3.1.0+incompatible h1:VDsGe6hquojHW/R/xMaaBvuoVkeLBm3yfQtt1yAruHA= -
 v3.2.0+incompatible h1:sRV2pLy3ylKn7EwwhwWubgvsGBJN1wxJmRJpGjtLXJE= -
 v3.2.1+incompatible h1:QM6Bh7E6CFJ+lSEHrq15zOyLiHgc3l8Mx8viTWus/v8= -
 v3.2.2+incompatible h1:nzw0p6dRxw0Xt2GGLKhnyIVFjkGlo0pqB9wFmeQnDTo= -
 v3.2.3+incompatible h1:TtqqdzcZfIWOXaXjEZLvz7Gui+4dASQkDcRG1lHf4tA= -
 v3.3.0+incompatible h1:vOAGOjJlc3w+PK6BNJyovQ1QxotdHwqlKg7OHWUmqx4= -
 v3.3.1+incompatible h1:LlSVhn3AwBL2SPmQ861Aw5clO9hfxkUMUCO28FN5PPU= -
 v3.4.0+incompatible h1:rPF2cldF1XQ9ubiSBWhxSuoyBPNlPOUMafzv73QzBEE= -
 v3.4.1+incompatible h1:jkXet0CDmdaMZctaF5qELIAFM7eeUx1nh3kMvLejAXk= -
github.com/sergi/go-diff
 v1.0.0 h1:Kpca3qRNrduNnOQeazBd0ysaKrUJiIuISHxogkT9RPQ= h1:0CfEIISq7TuYL3j771MWULgwwjU+GofnZX9QAmXWZgo=
github.com/sevlyar/go-daemon
 v0.1.0 h1:doVC2jeM2huo2NWnnBOu1W/JYTEvAI/T+6QkkizDjHQ= h1:6dJpPatBT9eUwM5VCw9Bt6CdX9Tk6UWvhW3MebLDRKE=
 v0.1.1 h1:2/2BKMKPWOdNxTId0dGweFB22w4I38HV1s4AQZTzJE0= -
 v0.1.2 h1:y0/6ymtNDw2yFHz9071ScpoEp6lRb3nF6FDwd6lMd2Q= -
 v0.1.3 h1:4yJ+cJwCwzoXwdhX1tJQ07ojr6Rdi8T2W4Rwel/85OI= -
 v0.1.4 h1:Ayxp/9SNHwPBjV+kKbnHl2ch6rhxTu08jfkGkoxgULQ= -
github.com/shirou/gopsutil
 v2.0.0+incompatible h1:dzbyDFOvUY4Mj7CyAz78DfKYUIL2KKHsm8BNs0koeqU= h1:5b4v6he4MtMOwMlS0TUMTu2PcXUg8+E1lC7eC3UO/RA=
 v2.16.10+incompatible h1:mgdmRYXgaBhMkqUwB8qbNw43NFuG6TvKLeNTqUUXB3M= -
 v2.16.11+incompatible h1:HGn85JG7+nt9NuQwbxd2zk1MVMJL+AFn2OhDpBhcTng= -
 v2.16.12+incompatible h1:9UOkm3qI9KvgTCLcays9oYepttUvJytUwSJIN2hkFkg= -
 v2.17.10+incompatible h1:kxelzrE1tvEmUMqHp2uP7LFmkDoViSwFEK2ebrO/SS0= -
 v2.17.11+incompatible h1:4L8jVLIf9tewjdy906qwjwUr3M9/ErnFv2qnznuH1E0= -
 v2.17.12+incompatible h1:FNbznluSK3DQggqiVw3wK/tFKJrKlLPBuQ+V8XkkCOc= -
 v2.18.10+incompatible h1:cy84jW6EVRPa5g9HAHrlbxMSIjBhDSX0OFYyMYminYs= -
 v2.18.11+incompatible h1:PMFTKnFTr/YTRW5rbLK4vWALV3a+IGXse5nvhSjztmg= -
 v2.18.12+incompatible h1:1eaJvGomDnH74/5cF4CTmTbLHAriGFsTZppLXDX93OM= -
github.com/sirupsen/logrus
 v1.0.0 h1:XM8X4m/9ACaclZMs946FQNEZBZafvToJLTR4007drwo= h1:pMByvHTf9Beacp5x1UXfOR9xyW/9antXMhjMPG0dEzc=
 v1.0.1 h1:k86xHae/+DCqiP5ac5Gf0AQtH+4mg0m6vE1os/WzmJ4= -
 v1.0.3 h1:B5C/igNWoiULof20pKfY4VntcIPqKuwEmoLZrabbUrc= -
 v1.0.4 h1:gzbtLsZC3Ic5PptoRG+kQj4L60qjK7H7XszrU163JNQ= -
 v1.0.5 h1:8c8b5uO0zS4X6RPl/sd1ENwSkIc0/H2PaHxE3udaE8I= -
 v1.0.6 h1:hcP1GmhGigz/O7h1WVUM5KklBp1JoNS9FggWKdj/j3s= -
 v1.1.0 h1:65VZabgUiV9ktjGM5nTq0+YurgTyX+YI2lSSfDjI+qU= h1:zrgwTnHtNr00buQ1vSptGe8m1f/BbgsPukg8qsT7A+A=
 v1.1.1 h1:VzGj7lhU7KEB9e9gMpAV/v5XT2NVSvLJhJLCWbnkgXg= -
 v1.2.0 h1:juTguoYk5qI21pwyTXY3B3Y5cOTH3ZUyZCg1v/mihuo= h1:LxeOpSwHxABJmUn/MG1IvRgCAasNZTLOkJPxbbu5VWo=
 v1.3.0 h1:hI/7Q+DtNZ2kINb6qt/lS+IyXnHQe9e90POfeewL/ME= -
github.com/sorcix/irc
 v1.0.0 h1:0uuxV8z/P6un0I0B0qr93zTjPhbeQcmfRRV4hUAjXNw= h1:MhzbySH63tDknqfvAAFK3ps/942g4z9EeJ/4lGgHyZc=
 v1.1.0 h1:YgP/7XL6gjGPxSScSjW5lv6/aYOA6HqlKlU57PFjwI4= -
 v1.1.1 h1:5JPmJFC/SxZl4OKFA9tnRRONCLI23vUYH9jsoYowK/Y= -
 v1.1.2 h1:aqNJ+9KmdjYarJHct96cLjSpFQovV/FQ8IfCVMNo2YY= -
 v1.1.3 h1:+ejCOdjCkSrRrekt+Ap28pa3XH1MwzmNluDu8UylWjY= -
 v1.1.4 h1:KDmVMPPzK4kbf3TQw1RsZAqTsh2JL9Zw69hYduX9Ykw= -
github.com/spf13/afero
 v1.0.0 h1:Z005C09nPzwTTsDRJCQBVnpTU0bjTr/NhyWLj1nSPP4= h1:j4pytiNVoe2o6bmDsKpLACNPDBIoEAkihy7loJ1B0CQ=
 v1.0.1 h1:iJfkwkEeXGckEoQWD8dLaygZWLXMolnndesiNuXfmKI= -
 v1.0.2 h1:5bRmqmInNmNFkI9NG9O0Xc/Lgl9wOWWUUA/O8XZqTCo= -
 v1.1.0 h1:bopulORc2JeYaxfHLvJa5NzxviA9PoWhpiiJkru7Ji4= -
 v1.1.1 h1:Lt3ihYMlE+lreX1GS4Qw4ZsNpYQLxIXKBTEOXm3nt6I= -
 v1.1.2 h1:m8/z1t7/fwjysjQRYbP0RD+bUIF/8tJwPdEZsI83ACI= -
 v1.2.0 h1:O9FblXGxoTc51M+cqr74Bm2Tmt4PvkA5iu/j8HrkNuY= h1:9ZxEEn6pIJ8Rxe320qSDBk6AsU0r9pR7Q4OcevTdifk=
 v1.2.1 h1:qgMbHoJbPbw579P+1zVY+6n4nIFuIchaIjzZ/I/Yq8M= -
github.com/spf13/cast
 v1.0.0 h1:GNbxZJxRIvehsqPCmvpb/fnBMMyMoF7lojcquQccV4k= h1:r2rcYCSwa1IExKTDiTfzaxqT2FNHs8hODu4LnUfgKEg=
 v1.1.0 h1:0Rhw4d6C8J9VPu6cjZLIhZ8+aAOHcDvGeKn+cq5Aq3k= -
 v1.2.0 h1:HHl1DSRbEQN2i8tJmtS6ViPyHx35+p51amrdsiTCrkg= -
 v1.3.0 h1:oget//CVOEoFewqQxwr0Ej5yjygnqGkvggSE/gB35Q8= h1:Qx5cxh0v+4UWYiBimWS+eyWzqEqokIECu5etghLkUJE=
github.com/spf13/cobra
 v0.0.1 h1:zZh3X5aZbdnoj+4XkaBxKfhO4ot82icYdhhREIAXIj8= h1:1l0Ry5zgKvJasoi3XT1TypsSe7PqH0Sj9dhYf7v3XqQ=
 v0.0.2 h1:NfkwRbgViGoyjBKsLI0QMDcuMnhM+SBg3T0cGfpvKDE= -
 v0.0.3 h1:ZlrZ4XsMRm04Fr5pSFxBgfND2EBVa1nLpiy1stUsX/8= -
github.com/spf13/jWalterWeatherman
 v1.0.0 h1:rJjb5zrWwg5oWAneAydR1Gb11NYrTejW67H8K/Hkes4= h1:cQK4TGJAtQXfYWX+Ddv3mKDzgVb68N+wFjFa4jdeBTo=
github.com/spf13/pflag
 v1.0.0 h1:oaPbdDe/x0UncahuwiPxW1GYJyilRAdsPnq3e1yaPcI= h1:DYY7MBk1bdzusC3SYhjObp+wFpr4gzcvqqNjLnInEg4=
 v1.0.1 h1:aCvUg6QPl3ibpQUxyLkrEkCHtPqYJL4x9AuhqVqFis4= -
 v1.0.2 h1:Fy0orTDgHdbnzHcsOgfCN4LtHf0ec3wwtiwJqwvf3Gc= -
 v1.0.3 h1:zPAT6CGy6wXeQ7NtTnaTerfKOsV6V6F8agHXFiazDkg= -
github.com/spf13/viper
 v1.0.0 h1:RUA/ghS2i64rlnn4ydTfblY8Og8QzcPtCcHvgMn+w/I= h1:A8kyI5cUJhb8N+3pkfONlcEcZbueH6nhAm0Fq7SrnBM=
 v1.0.1 h1:PkHLMQpuZ5I2RwtH1PhsAYIsP/oHy/8DXVSXKL2dLs4= -
 v1.0.2 h1:Ncr3ZIuJn322w2k1qmzXDnkLAdQMlJqBa9kfAH+irso= -
 v1.0.3 h1:z5LPUc2iz8VLT5Cw1UyrESG6FUUnOGecYGY08BLKSuc= -
 v1.1.0 h1:V7OZpY8i3C1x/pDmU0zNNlfVoDz112fSYvtWMjjS3f4= -
 v1.1.1 h1:/8JBRFO4eoHu1TmpsLgNBq1CQgRUg4GolYlEFieqJgo= -
 v1.2.0 h1:M4Rzxlu+RgU4pyBRKhKaVN1VeYOm8h2jgyXnAseDgCc= h1:P4AexN0a+C9tGAnUFNwDMYYZv3pjFuvmeiMyKRaNVlI=
 v1.2.1 h1:bIcUwXqLseLF3BDAZduuNfekWG87ibtFxi59Bq+oI9M= -
 v1.3.0 h1:cO6QlTTeK9RQDhFAbGLV5e3fHXbRpin/Gi8qfL4rdLk= h1:ZiWeW+zYFKm7srdB9IoDzzZXaJaI5eL9QjNiN/DMA2s=
 v1.3.1 h1:5+8j8FTpnFV4nEImW/ofkzEt8VoOiLXxdYIDsB73T38= -
github.com/square/go-jose
 v2.1.3+incompatible h1:54TSMwbLPKeEqy9wyz7sc1kOQNysg4RYZtXS4ZTh2Fc= h1:7MxpAF/1WTVUu8Am+T5kNy+t0902CaLWM4Z745MkOa8=
 v2.1.4+incompatible h1:ZMsS/lf1Yn7rwimhyaOTRYxLSZ8gqpMSZSNcstWyitw= -
 v2.1.5+incompatible h1:GR1DwlwenN17GOUzNJweLrufd32vF8h0+TALg//Sfjk= -
 v2.1.6+incompatible h1:vlXeaqoFPTA+Q4bWegkZ8PhPq6Ke9np2Bkyi09tMdQc= -
 v2.1.7+incompatible h1:4aqiRzL9x3yAPDY1mlY41VU+doHAKo2rAZ/rYNNmIN0= -
 v2.1.8+incompatible h1:Hbk1mGTCxbgWzS+/OT1NHGClCniCOA+enqqU81T9VOY= -
 v2.1.9+incompatible h1:Tbc5JPjizorD+Jd9tLz8PSUJHPAbmwn6qpmDQa9/eFs= -
 v2.2.0+incompatible h1:lzRTcrPIXvZGaBDawhPNSJx/ornyZDLgIzVii63HY6Y= -
 v2.2.1+incompatible h1:waq0gPzpxNDCKOAsTHDbCaCk3cqAlsZG6q6F2Gr6ri8= -
 v2.2.2+incompatible h1:HbXcU3MDamCIs1i4/HHqgf7v94SlKqRtZBD7CZ2xkbA= -
github.com/stathat/go
 v1.0.0 h1:HFIS5YkyaI6tXu7JXIRRZBLRvYstdNZm034zcCeaybI= h1:+9Eg2szqkcOGWv6gfheJmBBsmq9Qf5KDbzy8/aYYR0c=
github.com/stianeikeland/go-rpio
 v1.0.0 h1:9a5DDConuUBSBR4mYsr9dV0uAxGypybCumg0KR2o3fc= h1:Sh81rdJwD96E2wja2Gd7rrKM+XZ9LrwvN2w4IXrqLR8=
 v2.0.0+incompatible h1:8uxUOyN1L8TSNowMYnn42xbF8xdPtYZ5DgCyJNuuLzc= -
 v3.0.0+incompatible h1:AWDcfdnPPZ7Jikis6HNj0EgujMfB4t7oi4nZd5cIqko= -
 v4.2.0+incompatible h1:CUOlIxdJdT+H1obJPsmg8byu7jMSECLfAN9zynm5QGo= -
github.com/stretchr/graceful
 v1.2.6 h1:3LHHE/B32UcsQeVFe/8wd/21Id7EbgkhrPsGJyAHhXs= h1:IxdGAOTZueMKoBr3oJIzdeg5CCCXbHXfV44sLhfAXXI=
 v1.2.7 h1:MAu7w+pY1dovqNWPW7KlqnewhIJeF40dkf+J9VjBuQs= -
 v1.2.8 h1:q50eQeVoROi4Wnwd58DCESHM/LLdcz47wRpdfQ7NEQg= -
 v1.2.9 h1:phSl/mM1Zln2lPIgZrDT4df9YcfjFh3X5wJeTNmpDF0= -
 v1.2.10 h1:ZJbjeY1xHrdEDTzblLI0+2a5whh4lnnDwNEt2sinWmU= -
 v1.2.11 h1:UteEM/teV7C1EzabEs8oQwIfOemeC3wHawSKCT8YfIU= -
 v1.2.12 h1:HAPBO/FgGnDwcwBYzSGZtsWbkAmB3wTSpMm2VbmX9yc= -
 v1.2.13 h1:E60ISijhFt1izPcDoG0TZ7VljR2JM/pTBdqJYhzu+us= -
 v1.2.14 h1:x7EoVKCkYLJKjZqsi1uhi60rBz29xeEDbENV3pV/h2U= -
 v1.2.15 h1:vmXbwPGfe8bI6KkgmHry/P1Pk63bM3TDcfi+5mh+VHg= -
github.com/stretchr/objx
 v0.1.0 h1:4G4v2dO3VZwixGIRoQ5Lfboy6nUhCyYzaqnIAPPhYs4= h1:HFkY916IF+rwdDfMAkV7OtwuqBVzrE8GR6GFx+wExME=
 v0.1.1 h1:2vfRuCMp5sSVIDSqO8oNnWJq7mPa6KVP3iPIwFBuy8A= -
github.com/stretchr/testify
 v1.1.1 h1:/Box+ZZJaXnWRh0iQMXTpvCvCp4jJBdkbAUOqWmg/qI= h1:a8OnRcib4nhh0OaRAV+Yts87kKdq0PP7pXfy6kDkUVs=
 v1.1.2 h1:QFDOepAvHBWiCBkOcExyHwJmxDzp/jJvBL3X9KaAdRI= -
 v1.1.3 h1:76sIvNG1I8oBerx/MvuVHh5HBWBW7oxfsi3snKIsz5w= -
 v1.1.4 h1:ToftOQTytwshuOSj6bDSolVUa3GINfJP/fg3OkkOzQQ= -
 v1.2.0 h1:LThGCOvhuJic9Gyd1VBCkhyUXmO8vKaBFvBsJ2k03rg= -
 v1.2.1 h1:52QO5WkIUcHGIR7EnGagH88x1bUzqGXTC5/1bDTUQ7U= -
 v1.2.2 h1:bSDNvY7ZPG5RlJ8otE/7V6gMiyenm9RtJ7IUVIAoJ1w= -
 v1.3.0 h1:TivCn/peBQ7UY8ooIcPgZFpTNSz0Q2U6UrFlUfqbe0Q= h1:M5WIy9Dh21IEIfnGCwXGc5bZfKNJtfHm1UVUgZn+9EI=
github.com/stripe/stripe-go
 v46.0.0+incompatible h1:FeS6R4IM5UKP8AinLCsw4on0PDlIKekx3/0vNjskiII= h1:A1dQZmO/QypXmsL0T8axYZkSN/uA/T/A64pfKdBAMiY=
 v55.10.0+incompatible h1:trgrjJsFbzXlFXteZSHa5hKXCeWCbqQhaBPubhTVhCM= -
 v55.11.0+incompatible h1:1IB40Daw/FDlWJHW/TSbyWcWlqR06UOmK580dfatqAU= -
 v55.12.0+incompatible h1:L+dbKPjwkg0oIa06wWWL4jAM7BRIMsV7JEBGktyfVVo= -
 v55.13.0+incompatible h1:s0FoTKpEPN1WtDYytmoeaJhQ5m9V2WPdhk2o8S7r9+0= -
 v55.14.0+incompatible h1:5+JWn2dKhB6Ee7V3AKZgB7kR48Lb9hbfkFIlzCLbYIo= -
 v55.15.0+incompatible h1:b2B5jgYWngtN5itn97gOV0oMR2zv6+uHmo0dqXbjC7g= -
 v56.0.0+incompatible h1:Z4pwj/CTNBbcuOvCmMxPV6iuQmFjqlfpFfnG/Lo0oCw= -
 v56.1.0+incompatible h1:UnP4XYnamOiBxSEk2jozw/zshI1OiOeHV4cOPYBR8O0= -
 v57.0.1+incompatible h1:vi1rMn1PoX2I1A6LClIUP7yuqgXnLt7lCnLalS/XFrs= -
github.com/tchap/go-patricia
 v2.0.0+incompatible h1:s7Y2vY0mGk1u/IErHsUivFRhkb1jSWP09JyaSbnb92o= h1:bmLyhP68RS6kStMGxByiQ23RP/odRBOTVjwp2cDyi6I=
 v2.1.0+incompatible h1:QYlSsoKB8eZ/WNANRnx6b+F13HMTCpPw3WMym3LZ4WM= -
 v2.2.0+incompatible h1:ak63y+w/aM8O4hlbxFx5PgHvyNn0YwMupPzZ21RPdfI= -
 v2.2.1+incompatible h1:8cGLPH4wLZBD60bSEeviBryfbiOjVWZZWWAkPuM3RKg= -
 v2.2.2+incompatible h1:4RD9WWPHfkjwFbXOa+XMIdY2KewZgsKOOwHBxfB3+uA= -
 v2.2.3+incompatible h1:MitC8FhU/PCgkw5ZWu/LgLYS/waSdOt3xF86soUwXJ0= -
 v2.2.4+incompatible h1:z74O1VZTf7PHImUDArMqC2IiZlsGRZnkUjLwzucXZo4= -
 v2.2.5+incompatible h1:3N7XRawYGEWcnymqXHvED3gn+/G46E+djx5co1/XZ/Y= -
 v2.2.6+incompatible h1:JvoDL7JSoIP2HDE8AbDH3zC8QBPxmzYe32HHy5yQ+Ck= -
 v2.3.0+incompatible h1:GkY4dP3cEfEASBPPkWd+AmjYxhmDkqO9/zg7R0lSQRs= -
github.com/tcnksm/go-gitconfig
 v0.1.1 h1:VC+UGqwt8FUGsp3Gz/Z7mgY4gbNVAhNRdocj/ALgci0= h1:/8EhP4H7oJZdIPyT+/UIsG87kTzrzM4UsLGSItWYCpE=
 v0.1.2 h1:iiDhRitByXAEyjgBqsKi9QU4o2TNtv9kPP3RgPgXBPw= -
github.com/tdewolff/minify
 v2.0.0+incompatible h1:QkVWcRNKCZqBCyQmi2eJOQYZabRCot3TIsJCyWm9NEo= h1:9Ov578KJUmAWpS6NeZwRZyT56Uf6o3Mcz9CEsg8USYs=
 v2.1.0+incompatible h1:NE/tjyBKNwfd3BzrkrFt2aRLTMjqcyT4ys6cJcZxU+g= -
 v2.2.0+incompatible h1:13L/LpBCvmlmsfgrARlJUhLyJ/2LnHUPTfNqDq8do4Y= -
 v2.3.0+incompatible h1:Tr6ipbiX6tZqFZDif2e/y3UGZqnGVxpflLeDKj+dAqw= -
 v2.3.1+incompatible h1:HW0RtdgW4ZBtx4RIbJiFrsYt8sWAPksAMPJeMP+Ocac= -
 v2.3.2+incompatible h1:fDO06OlGbj/J1xGMM4wqfkwmIt9RSunjda2Uyuw9RUg= -
 v2.3.3+incompatible h1:PE7SIX0z/lKHnNxWBiWbtd4hnYE6Wxk14nXpCRhYGOo= -
 v2.3.4+incompatible h1:kinygdLIU2uv48NlOtMTtlVOVgoPAt61MgPfxKHAK5s= -
 v2.3.5+incompatible h1:oFxBKxTIY1F/1DEJhLeh/T507W56JqZtWVrawFcdadI= -
 v2.3.6+incompatible h1:2hw5/9ZvxhWLvBUnHE06gElGYz+Jv9R4Eys0XUzItYo= -
github.com/tealeg/xlsx
 v1.0.0 h1:h90Zg7jJK4UcmuvrHBPe0Gsc+kKPc6qvKf0bdktVRbk= h1:uxu5UY2ovkuRPWKQ8Q7JG0JbSivrISjdPzZQKeo74mA=
 v1.0.1 h1:pwK27zRp12IRTFGGz/Vng4lyeEltxf1wVe33dR4bwFg= -
 v1.0.2 h1:BhIi9z8dVAhNbFb+4CNKZLQNoq8JLTxtwoicSAwqbTo= -
 v1.0.3 h1:BXsDIQYBPq2HgbwUxrsVXIrnO0BDxmsdUfHSfvwfBuQ= -
github.com/tedsuo/rata
 v1.0.0 h1:Sf9aZrYy6ElSTncjnGkyC2yuVvz5YJetBIUKJ4CmeKE= h1:X47ELzhOoLbfFIY0Cql9P6yo3Cdwf2CMX3FVZxRzJPc=
github.com/tidwall/gjson
 v1.0.6 h1:HYH2srVkCC53232wJXWptchlbF1n2JxJBmM8i1/bLXU= h1:c/nTNbUr0E0OrXEhq1pwa8iEgc2DOt4ZZqAt1HtCkPA=
 v1.1.0 h1:/7OBSUzFP8NhuzLlHg0vETJrRL02C++0ql5uSY3DITs= -
 v1.1.1 h1:XSn7wxSH2Us55nigCfI8WrNfe2gihrwOSJU39w7Ot2w= -
 v1.1.2 h1:2cScOmQ0oRDK1idscWbg9Va8xvQ88Lqb73rkgg8scEo= -
 v1.1.3 h1:u4mspaByxY+Qk4U1QYYVzGFI8qxN/3jtEV0ZDb2vRic= -
 v1.1.4 h1:lonRDhK9sFzw7ogkERBgx5wF6lRP2bpjr6jiwVzYjYc= -
 v1.1.5 h1:QysILxBeUEY3GTLA0fQVgkQG1zme8NxGvhh2SSqWNwI= -
 v1.1.6 h1:2USkZlXVqQJQpbUVkozothVPA9M/U7X7j42u9V8o6kA= -
 v1.2.0 h1:pSOXpbajgrqYACThJdSVg1XwMv/xUV8CfQHd3Ti6gQw= -
 v1.2.1 h1:j0efZLrZUvNerEf6xqoi0NjWMK5YlLrR7Guo/dxY174= -
github.com/tinylib/msgp
 v1.0.1 h1:iF+TMfZ81pSM9FEl47U+sg1cE6x7TZC+mPwqkXrIMSE= h1:+d+yLhGm8mzTaHzB+wgMYrodPfmZrzkirds8fDWklFE=
 v1.0.2 h1:DfdQrzQa7Yh2es9SuLkixqxuXS2SxsdYn0KbdrOGWD8= -
 v1.1.0 h1:9fQd+ICuRIu/ue4vxJZu6/LzxN0HwMds2nq/0cFvxHU= -
github.com/toqueteos/webbrowser
 v1.0.0 h1:RuypZ2eTUNrYG5WKWj4eU1ZEbxDPOuImkSADGfdsdNE= h1:Hqqqmzj8AHn+VlZyVjaRWY20i25hoOZGAABCcg2el4A=
 v1.1.0 h1:Prj1okiysRgHPoe3B1bOIVxcv+UuSt525BDQmR5W0x0= -
github.com/twinj/uuid
 v0.1.0 h1:m4NyLH5MReF8zWQegEXvSTpnV6A7svVLG5FbZql6OAA= h1:mMgcE1RHFUFqe5AfiwlINXisXfDGro23fWdPUfOMjRY=
 v1.0.0 h1:fzz7COZnDrXGTAOHGuUGYd6sG+JMq+AoE7+Jlu0przk= -
github.com/tylerb/graceful
 v1.2.6 h1:zAEMFDuukWHFPIa4EW2pcQfXfUhXC868lWH37njzYVk= h1:LPYTbOYmUTdabwRt0TGhLllQ0MUNbs0Y5q1WXJOI9II=
 v1.2.7 h1:zEMOuHRwei9u0s8MoZoP9ghxsLS88bUnrI1KWZydKS0= -
 v1.2.8 h1:grDNUs6WOQUa14aJT24nl3BZr1DQACw/9k/PDRTkjcE= -
 v1.2.9 h1:DOgh8k6CAtzRgteq4mh4HUOLrBJBLRh8txtlJOmMccE= -
 v1.2.10 h1:/7izbl8iDZiY8xt74IWoH9xBP6sb+3wG4z+QvPbbjCM= -
 v1.2.11 h1:mLjyFVXDHsDZqOv6Y8+CtjHzGUcE7wNnYU6RwOS5HdY= -
 v1.2.12 h1:NQ2YbMiJEFNuYoqiXr4VffRGjuKePwGQkgYA/rjXcZg= -
 v1.2.13 h1:yKdTh6eHcWdD8Jm3wxgJ6pNf8Lb3wwbV4Ip8fHbeMLE= -
 v1.2.14 h1:QRmtSwCR4sMTRSfXPx+P08QdrdED91nyJqOieBOuYuc= -
 v1.2.15 h1:B0x01Y8fsJpogzZTkDg6BDi6eMf03s01lEKGdrv83oA= -
github.com/uber-go/zap
 v1.3.0 h1:RRVQvokMHcJTzc6dkeCp5op1ciD8xPp/2C4QOJkOC6I= h1:GY+83l3yxBcBw2kmHu/sAWwItnTn+ynxHCRo+WiIQOY=
 v1.4.0 h1:X8dxy6V7E4ET+6Wd+ZVfAAimB59jZDq9FBZhP+RsUjE= -
 v1.4.1 h1:vhSW6UUNKbKI+nxkFOf56RzwpmIFpJryQDpKLeEN7jU= -
 v1.5.0 h1:EmvaEjplILc2Vl1CuES8hUX3SHCtWhkbseSpLdDI8lE= -
 v1.6.0 h1:eqi3eQ2sY6ysJMmF7pvs7iyq0Dp4yTSw+6Jj+5tSGOE= -
 v1.7.0 h1:BYRQpyYN6nFkUkexxDiSViXs27cT+Uukqr18NY7aRxw= -
 v1.7.1 h1:LvIn2hjGWsKJ4khx5Ornjo1pj5HDkAsAf17a+uIoD9g= -
 v1.8.0 h1:XZzjHgXBPsLh1Ndfv0QZnPskh54Z8ZD641b+Ka0Ey9c= -
 v1.9.0 h1:HSDaxzxZlbVzACpHSNfxOga0D/vTBAuE7vUarwRGeGg= -
 v1.9.1 h1:CZN7Pmty0PLtqZEi3N8VSI0Us8b0RL5ah6l1jOH3ZJ0= -
github.com/uber/jaeger-client-go
 v2.8.0+incompatible h1:7DGH8Hqk6PirD+GE+bvCf0cLnspLuae7N1NcwMeQcyg= h1:WVhlPFC8FDjOFMMWRy2pZqQJSXxYSwNYOkTr/Z6d3Kk=
 v2.9.0+incompatible h1:40VHY+zJx7cq8hZNvlgfCkGaWO06oipmsZ4M1C0Qsfs= -
 v2.10.0+incompatible h1:P5Svqhsg48/Hpa0syPMD8hy6QS8eRZfXEbJDqHw4b6U= -
 v2.11.0+incompatible h1:NEBBje7vFQeIfc529syQaeYUrlNQ6EzoiZlbw2SDkEw= -
 v2.11.1+incompatible h1:hDP5AohdNCS0V8Qt9UOHl1hRthg7wE4y4lpwXB2/DPc= -
 v2.11.2+incompatible h1:D2idO5gYBl+40qnsowJaqtwCV6z1rxYy2yhYBh3mVvI= -
 v2.12.0+incompatible h1:byY6dnhKNNpznX1J2JbAYIQpXJqhWM0pt2opp/7Ug1s= -
 v2.13.0+incompatible h1:BQ7GxyS54wK+5kfRNoMVOhgQ7VAjhYlFq4rAhV7pnHc= -
 v2.14.0+incompatible h1:1KGTNRby0tDiVDDhvzL0pz0N26M9DobVCfSqz4Z/UPc= -
 v2.15.0+incompatible h1:NP3qsSqNxh8VYr956ur1N/1C1PjvOJnJykCzcD5QHbk= -
github.com/ugorji/go
 v1.1.1 h1:gmervu+jDMvXTbcHQ0pd2wee85nEoE0BsVyEuzkfK8w= h1:hnLbHMwcvSihnDhEfx2/BzKp2xb0Y+ErdfYcrs9tkJQ=
 v1.1.2 h1:JON3E2/GPW2iDNGoSAusl1KDf5TRQ8k8q7Tp097pZGs= -
github.com/unrolled/render
 v1.0.0 h1:XYtvhA3UkpB7PqkvhUFYmpKD55OudoIeygcfus4vcd4= h1:tu82oB5W2ykJRVioYsB+IQKcft7ryBr7w12qMBUPyXg=
github.com/unrolled/secure
 v1.0.0 h1:2p4MlT30bNNjaFxA+gtDuLT/73fnXblTC+W/lCzOaZc= h1:mnPT77IAdsi/kV7+Es7y+pXALeV3h7G6dQF6mNYjcLA=
github.com/urfave/cli
 v1.15.0 h1:mPJ+IrlccdnKDrlo72DWcMcbjAHVt9bGwI8Cx8wsxO4= h1:70zkFmudgCuE/ngEzBv17Jvp/497gISqfk5gWijbERA=
 v1.16.0 h1:0MZKChuQWAfo0FuyKUMdR3kCdTrCANLMuo5SXd3xV2k= -
 v1.16.1 h1:NP9yyLRSRvF1/+CeEs9A0U2vzJb4JvQ3GIid09BgdDk= -
 v1.17.0 h1:pxxCYHgqkpprfAbC/5DlBd/n5TQkQpEY/qwERAr4vkE= -
 v1.17.1 h1:QSJ6c7oou2nXchu1zV0ZSwC2YI0o3i20OvvQyl2yfwE= -
 v1.18.0 h1:m9MfmZWX7bwr9kUcs/Asr95j0IVXzGNNc+/5ku2m26Q= -
 v1.18.1 h1:IFc93MpteseEF1dvLEwx5Zn+K7xkbIcBGp36OwYlFx8= -
 v1.19.0 h1:dBWCicHK8GorrWSPcMemx1MwzvW2m4rSVEAvUrCtKpw= -
 v1.19.1 h1:0mKm4ZoB74PxYmZVua162y1dGt1qc10MyymYRBf3lb8= -
 v1.20.0 h1:fDqGv3UG/4jbVl/QkFwEdddtEDjh/5Ov6X+0B/3bPaw= -
github.com/urfave/negroni
 v0.1.0 h1:I0ouxPWkMNjhNUCCxaS2xMCcM1sbwMcNmAkBjYBNRhk= h1:Meg73S6kFm/4PpbYdq35yYWoCZ9mS/YSx+lKnmiohz4=
 v0.2.0 h1:cadBY8/+9L/dTagBqV7N0l/SJiB4Wg+os5QdmaFY5Wg= -
 v0.3.0 h1:PaXOb61mWeZJxc1Ji2xJjpVg9QfPo0rrB+lHyBxGNSU= -
 v1.0.0 h1:kIimOitoypq34K7TG7DUaJ9kq/N4Ofuwi1sjz0KipXc= -
github.com/valyala/bytebufferpool
 v1.0.0 h1:GqA5TC/0021Y/b9FG4Oi9Mr3q7XYx6KllzawFIhcdPw= h1:6bBcMArwyJ5K/AmCkWv1jt77kVWyCJ6HpOuEn7z0Csc=
github.com/valyala/fasthttp
 v1.0.0 h1:BwIoZQbBsTo3v2F5lz5Oy3TlTq4wLKTLV260EVTEWco= h1:4vX61m6KN+xDduDNwXrhIAVZaZaZiQ1luJk8LWSxF3s=
 v1.1.0 h1:3BohG7mqwj4lq7PTX//7gLbUlzNvZSPmuHFnloXT0lw= -
 v1.2.0 h1:dzZJf2IuMiclVjdw0kkT+f9u4YdrapbNyGAN47E/qnk= -
github.com/vbatts/tar-split
 v0.9.9 h1:Gudkuvaj/F5kpyLskuy6dFopZvPyLqAf/BG/3dznhGw= h1:LEuURwDEiWjRjwu46yU3KVGuUdVv/dcnpcEPSzR8z6g=
 v0.9.10 h1:ekzO3fGq/l9APJgmGtPHV9Kf8FJMASoupdFMCtEoxLs= -
 v0.9.11 h1:zKLOYouE7iQtZjf96FTrxyw/F2+2UxautE3z7ylfs8E= -
 v0.9.12 h1:1CsW7Z93nGZtrM0Jyi1nbrNMx6QizAXIhWqOfc/XlTo= -
 v0.9.13 h1:g/A49fgfpiRpav/tNBaDFF2X1q2OLKUzxp+08eThd0w= -
 v0.10.0 h1:YNiiibPBopDxqMmwpHl9MF+qHr79G+ktuHR769gCtcU= -
 v0.10.1 h1:eSmfbYDBO+qCIvHPMNgmkIXbf5HUO2UYSd/+Z11zHs0= -
 v0.10.2 h1:CXd7HEKGkTLjBMinpObcJZU5Hm8EKlor2a1JtX6msXQ= -
 v0.11.0 h1:Vdj/+9462ZtnhhfQylOpk4xC4IqMui1JSgdIbucTOLw= -
 v0.11.1 h1:0Odu65rhcZ3JZaPHxl7tCI3V/C/Q9Zf82UFravl02dE= -
github.com/vdemeester/shakers
 v0.1.0 h1:K+n9sSyUCg2ywmZkv+3c7vsYZfivcfKhMh8kRxCrONM= h1:IZ1HHynUOQt32iQ3rvAeVddXLd19h/6LWiKsh9RZtAQ=
github.com/veandco/go-sdl2
 v0.1.0 h1:+mM3KPG4mVsogWe3JRVCFg2B4TemV24U4tAN0IwsGYs= h1:FB+kTpX9YTE+urhYiClnRzpOXbiWgaU3+5F2AB78DPg=
 v0.2.0 h1:+T0uGDteCqkfSGMlvjKPW60vt1IOyFGX549WHK7HX9U= -
 v0.3.0 h1:IWYkHMp8V3v37NsKjszln8FFnX2+ab0538J371t+rss= -
github.com/vishvananda/netlink
 v1.0.0 h1:bqNY2lgheFIu1meHUFSH3d7vG93AFyqg3oGbJCOJgSM= h1:+SR5DhBJrl6ZM7CoCKvpw5BKroDKQ+PJqOg65H/2ktk=
github.com/vmihailenco/msgpack
 v3.2.7+incompatible h1:XKm+r81VgMa2OFKG02gtj5fJQT8u8XV7GoITscnjXA0= h1:fy3FlTQTDXWkZ7Bh6AcGMlsjHatGryHQYUTf1ShIgkk=
 v3.2.8+incompatible h1:RTze2mAXYgUzv/I3Aa4294R7Ou2sR2yv+V1rd0MzjSs= -
 v3.2.9+incompatible h1:mRCad8j+rpoOgFYpWT2az2f9KOCohRwgc4qaLZKhIwo= -
 v3.3.0+incompatible h1:VOLxk/ZoLVBhRSLNJLGyf+okTQNz/8D7G115oshWw4w= -
 v3.3.1+incompatible h1:ibe+d1lqocBmxbJ+gwcDO8LpAHFr3PGDYovoURuTVGk= -
 v3.3.2+incompatible h1:6Y7b3m/E53cNMOoEPeq5QeXdzc4IThHxPXt3jJM0EQk= -
 v3.3.3+incompatible h1:wapg9xDUZDzGCNFlwc5SqI1rvcciqcxEHac4CYj89xI= -
 v4.0.0+incompatible h1:R/ftCULcY/r0SLpalySUSd8QV4fVABi/h0D/IjlYJzg= -
 v4.0.1+incompatible h1:RMF1enSPeKTlXrXdOcqjFUElywVZjjC6pqse21bKbEU= -
 v4.0.2+incompatible h1:6ujmmycMfB62Mwv2N4atpnf8CKLSzhgodqMenpELKIQ= -
github.com/vmware/govcloudair
 v0.0.1 h1:7Y+0Lhjokx2t4x44lm0mGZC7tR5NSOpi5nEB37Hf3qw= h1:Vxktpba+eP4dX5YzYP869DRPSm5ChQ2A/GUrmKSLvlo=
 v0.0.2 h1:ki01OjlgpEWyEc7iZTTaWW9tISSWafiqj/PHLPB4Iwc= -
github.com/vmware/govmomi
 v0.12.1 h1:AjQ70q+b3j8YVuxkaEWSHpoybZyFLN/Kmvs6wUJBzdU= h1:URlwyTFZX72RmxtxuaFL2Uj3fD1JTvZdx59bHWk6aFU=
 v0.13.0 h1:1KhlQeH2rSlYcIQJKY2GGd7HW4np1N2HHHPuOsCBS6w= -
 v0.14.0 h1:OzYabeN/Ex4PkqhQf+HXUnRoRsNBpjmSJO/cusWkeNw= -
 v0.15.0 h1:fVMjwFASkUIGenwURwP0ruAzTjka0l2AV9wtARwkJLI= -
 v0.16.0 h1:4tGQtzkzcA/4JIcZ0OpqBnz+q8TFA5FeYCzaJyZJH38= -
 v0.17.0 h1:AxhHt5FCuWjaiBg+Yh77wN2DxcO7y09NbqZ8sACCS0o= -
 v0.17.1 h1:ZaFC7mIp7W5VZaTQPklLn7cJVEP4EX3XUYP0ler5l80= -
 v0.18.0 h1:f7QxSmP7meCtoAmiKZogvVbLInT+CZx6Px6K5rYsJZo= -
 v0.19.0 h1:CR6tEByWCPOnRoRyhLzuHaU+6o2ybF3qufNRWS/MGrY= -
 v0.20.0 h1:+1IyhvoVb5JET2Wvgw9J3ZDv6CK4sxzUunpH8LhQqm4= -
github.com/vulcand/vulcand
 v0.8.0-alpha h1:QpVjeMrMNPA8g+eB7sLXStZpKUNVhftEs+iY9Az/mJM= h1:VPQyjgDrzUJfiBz1sLRYBrlkXsGW8VxIDhGrMeGNUXE=
 v0.8.0-alpha.1 h1:wlwF2NTTgZDlE0bmm+wZxCe07YPpg9h8FQWVzj3VM48= -
 v0.8.0-alpha.2 h1:Zd4gRQcTFzwLyyuIYsTmCs/CRr6IwyOQcFj9ACFxD+k= -
 v0.8.0-alpha.3 h1:c5Oxy7OM/KDzqB7UJaCUHIe/Y5DT4gjg3KfyZqArEoU= -
 v0.8.0-alpha.4 h1:/ukypwpt4vPs7qsucUQ0xLCFQtQgIIZOR/pOuD7o4jM= -
 v0.8.0-beta.1 h1:82W1bUtCYQweZncsCfnvdYEZLNI4u1/K8Tcpk+nzz74= -
 v0.8.0-beta.2 h1:8EVUgb3ITiCySftJGt2CAnCl3OepBiobdEGmzvk4dm4= -
 v0.8.0-beta.3 h1:PSWnGcpVCmNKy02GQDTdJUxJlE2AmxJ+ndKogSQrpAY= -
github.com/willf/bitset
 v1.0.0 h1:sahTKpJ5zuV/IIwoWOTMyXru5uMFdujTzqURICXBI+g= h1:RjeCKbqT1RxIR/KWY6phxZiaY1IyutSBfGjNPySAYV4=
 v1.1.2 h1:qRQzojujJ9p4JrdmSxeu3hn348shKWovBYAQth9NoTg= -
 v1.1.3 h1:ekJIKh6+YbUIVt9DfNbkR5d6aFcFTLDRyJNAACURBg8= -
 v1.1.4 h1:G66e8XEJ38EupTBD2OJy1/0YJnM5txRThGM6BERYgpw= -
 v1.1.5 h1:YacgXd1Q7p+U4L65olNx/aHZvDH/LENZdxJFlTXZHX4= -
 v1.1.6 h1:z+P76zpmyqQwYPRnR4Cpmd0EHNlZjpbFMg4CyvS5WOs= -
 v1.1.7 h1:Ox3YCFMSJlmHyn6nTVi5Xo6u4ayVEGKq1UyHnBEeFFs= -
 v1.1.8 h1:dkYSYYvk7YN94DtrGQEp8CZOb/jNlMnH9+FkgFRSDek= -
 v1.1.9 h1:GBtFynGY9ZWZmEC9sWuu41/7VBXPFCOAbCbqTflOg9c= -
 v1.1.10 h1:NotGKqX0KwQ72NUzqrjZq5ipPNDQex9lo3WpaS8L2sc= -
github.com/willf/bloom
 v1.0.0 h1:D857mzMaGe1NU89jxegrN7Feh44FOQqbwzkMbMBdNz0= h1:MmAltL9pDMNTrvUkxdg0k0q5I0suxmuwp3KbyrZLOZ8=
 v2.0.0+incompatible h1:MfZ2AYvCUeCeJI1/UZ32xGxB/WHEvIZXZVwtki+J618= -
 v2.0.1+incompatible h1:Px+FzUVwNJIDZZuL6qhTqsEtRCbLvSN7+ePQIrhr0EI= -
 v2.0.2+incompatible h1:bbvr/JeHpMxAFKmODoHMayEx6hBjNwO5nu12oWiiJOI= -
 v2.0.3+incompatible h1:QDacWdqcAUI1MPOwIQZRy9kOR7yxfyEmxX8Wdm2/JPA= -
github.com/x-cray/logrus-prefixed-formatter
 v0.1.0 h1:vO0zTDdZpdndAWlW827FTo5Wj/Ve+oksPzVBujy9gZw= h1:2duySbKsL6M18s5GU7VPsoEPHyzalCE06qoARUCeBBE=
 v0.2.0 h1:fKTfWYeL7NYB25WG5gLylnow2oFJFTd+hagnLTcG8dI= -
 v0.3.0 h1:kVa69uBpEeq88pPVvbVmXn5ryPTHnWICNigQBfn2ax4= -
 v0.3.1 h1:jZQgJCBN4Yigi9Q649z5rjZmmG/UMjuCK2gYLMR9IX8= -
 v0.3.2 h1:kgW77M6M5sKr3Ux8ypddC11Rpwzr28nfR2GPZq+6EZU= -
 v0.4.0 h1:c13Xj1qeaujIEJdini0xFd6fepU8xsAnY3346+/5CMQ= -
 v0.5.0 h1:WLnhI5ksPAxJmGY/KmlymY3MmRY9xVJ6JLQU2PzitYg= -
 v0.5.1 h1:eG49gUCQh30PLZrITtyuetb6PJK49Fr3kIT2wvmYWIw= -
 v0.5.2 h1:00txxvfBM9muc0jiLIEAkAcIMJzfthRT6usrui8uGmg= -
github.com/xanzy/go-cloudstack
 v2.1.7+incompatible h1:PHybkWJkFodL0bCEl3Jq7cj5jROGPgza7HYoVbGWPe4= h1:s3eL3z5pNXF5FVybcT+LIVdId8pYn709yv6v5mrkrQE=
 v2.2.0+incompatible h1:aVsJ0WcCW3kAPpSTuXUZI3Ka918G2FIr0VzxyrCykRc= -
 v2.2.1+incompatible h1:pyYD8gqHH5z+xf7/Qrw33DQPbuQ6P9g0KTetc9+4OV4= -
 v2.3.0+incompatible h1:n6LDLHAWaYlGb+cxgOtmFSDi9Df5iQ6nNE2s9u8KKnQ= -
 v2.3.1+incompatible h1:7pHGnNozkulIsB1D8oQ/4Hqb3NVO/3iCIixBt3YJW94= -
 v2.3.2+incompatible h1:Al0eyKWmKzPCIiXbnE7lj18RN1HQTyhoLU8MId/UJh4= -
 v2.3.3+incompatible h1:bvQueLtgVICiSHLJyGbqYTcH5vdFgsZoItxgd+tAyK8= -
 v2.3.4+incompatible h1:37K+uzJT6GshLkBhgbDEfMnn/0ww+Fy+kIOBn0zN9ZI= -
 v2.4.0+incompatible h1:unsedy0PbzPl3k1F7ZEV2fONdtpAdDxyIjJhSbhzUmM= -
 v2.4.1+incompatible h1:Oc4xa2+I94h1g/QJ+nHoq597nJz2KXzxuQx/weOx0AU= -
github.com/xanzy/ssh-agent
 v0.1.0 h1:lOhdXLxtmYjaHc76ZtNmJWPg948y/RnT+3N3cvKWFzY= h1:0NyE30eGUDliuLEHJgYte/zncp2zdTStcOnWhgSqHD8=
 v0.2.0 h1:Adglfbi5p9Z0BmK2oKU9nTG+zKfniSfnaMYB+ULd+Ro= -
github.com/xeipuuv/gojsonschema
 v1.1.0 h1:ngVtJC9TY/lg0AA/1k48FYhBrhRoFlEmWzsehpNAaZg= h1:5yf86TLmAcydyeJq5YvxkGPE2fm/u4myDekKRoLuqhs=
github.com/xenolf/lego
 v0.5.0 h1:TmqIS0KL2EM61cH1oJkHy2WOA966yBCpMlAoGfGZx6A= h1:fwiGnfsIjG7OHPfOvgK7Y/Qo6+2Ox0iozjNTkZICKbY=
 v1.0.0 h1:NGBQjovPXKQKlVIvBkHSZ8CWryqokHohSpeaU0U89ss= -
 v1.0.1 h1:Rr9iqO8MoNxY6OvqdIZTnNZ8bwt0RNz00nGXfoTq4Bc= -
 v1.1.0 h1:Ias1pE9hO98/fI23RLza0T3461YiM720d96oxTRPyuM= -
 v1.2.0 h1:oeYRLMzAESnUzbQQFx+ma17Cnd3Vv05s+jAwFhmm6lo= -
 v1.2.1 h1:wAsBCIaTDlgYbR/yVuP0gzcnZrA94NVc84K6vfOIyyA= -
 v2.0.0+incompatible h1:zAuEuejarWN0l2DJUqQj5qtuhwdvw3a5gH9g0nmOaIY= -
 v2.0.1+incompatible h1:5x9Zy8MXq0zVvssH5Jk4b3kbDUe5voX1qGTfKA72HAk= -
 v2.1.0+incompatible h1:zZErna+4KHeBsUC3mw6gthaXncPDoBuFJOHKCRl64Wg= -
 v2.2.0+incompatible h1:r4UAcpgPmX3j0aThoVrRM1FFLcvyy08UyGbIwFU4zoQ= -
github.com/xordataexchange/crypt
 v0.0.1 h1:x1PDSf0nqx2pbejrf5Iy6Dli1MVt++YbvScg0tCtPq4= h1:aYKd//L2LvnjZzWKhF00oedf4jCCReLcmhLdhm1A27Q=
 v0.0.2 h1:VBfFXTpEwLq2hzs42qCHOyKw5AqEm9DYGqBuINmzUZY= -
github.com/yosssi/ace
 v0.0.1 h1:yYB7ieTnrxWWtwXC/MyHMd9HbpWptXYEYzek8nnrvEQ= h1:ALfIzm2vT7t5ZE7uoIZqF3TQ7SAOyupFZnkrF5id+K0=
 v0.0.2 h1:IL7wOYswDn5VvzrBemHhn8AyQXzkuSMX4XR0vO503c4= -
 v0.0.3 h1:HTUFLROktpQdsdvjw4brIkQoBN115DA7aF2qRt8b0e4= -
 v0.0.4 h1:7Mp+sPgLVDHF3w7lDPpfj4ZCTeij6XKFGwGjUwW4mAs= -
 v0.0.5 h1:tUkIP/BLdKqrlrPwcmH0shwEEhTRHoGnc1wFIWmaBUA= -
github.com/youtube/vitess
 v2.0.0+incompatible h1:uYXSuBGF6jKqICshaTvE8qLKYK6HUjRLlWF1ofa2Ttw= h1:hpMim5/30F1r+0P8GGtB29d0gWHr0IZ5unS+CG0zMx8=
 v2.1.0-alpha.1+incompatible h1:A1S51tG+yZ4etVTr/Jm6E79iVuIMxhSV5Vs108dHDXE= -
 v2.1.0-alpha.2+incompatible h1:BoehBgBa/N8lJpNxEopdssbx44VERvbqfGw0CEJR8F8= -
 v2.1.0-rc.1+incompatible h1:IOx7u+NfFJCX/eO9opZ0YnoM4q3EtRfM0NlTbiZ7JI8= -
 v2.1.0+incompatible h1:yNlNxqFSbes4v+CVlFrW8PV/qt4LlMGR4IqWcNN6DMY= -
 v2.1.1+incompatible h1:SE+P7DNX/jw5RHFs5CHRhZQjq402EJFCD33JhzQMdDw= -
 v2.2.0-rc.1+incompatible h1:3v09CMupxKooCj60rwxVrSDYnWy+Vrm++Vr3AX4R/5M= -
 v3.0.0-rc.1+incompatible h1:BXgQz4Q1DxL5t0V69nJyIf6lLstae0IDePOR9XdwKAM= -
 v3.0.0-rc.2+incompatible h1:ni30FAc3SPToUdH6fPX6+bKLe6gZETioFtWvRIbVhQM= -
 v3.0.0-rc.3+incompatible h1:+mxAImN50PmcSt39GwG08nmjVvFL+arNbv1pxUxaG0s= -
github.com/yvasiyarov/gorelic
 v0.0.1 h1:tQfsTzdBJh7SACDKgBj2DomO0BsNGPu4bnSQayou5GU= h1:NUSPSUX/bi6SeDMUh6brw0nXpxHnc96TguQh0+r/ssA=
 v0.0.2 h1:vkBwl2sn50/KY9cgiJ0qiew4RTtlE4lQ1wtxY5aEKW0= -
 v0.0.4 h1:WxTLIZSxJ4bin7nwHPDClEJODZjBnr7yKupw2Mi/9rE= -
 v0.0.5 h1:Kx2Uz/YBHKb0HUxw4x1jGf+vdo2VOlRS+2v8N6zjVIw= -
 v0.0.6 h1:qMJQYPNdtJ7UNYHjX38KXZtltKTqimMuoQjNnSVIuJg= -
github.com/zeebo/bencode
 v1.0.0 h1:zgop0Wu1nu4IexAZeCZ5qbsjU4O1vMrfCrVgUjbHVuA= h1:Ct7CkrWIQuLWAy9M3atFHYq4kG9Ao/SsY5cdtCXmp9Y=
github.com/zenazn/goji
 v0.8.1 h1:EeiWCsQlqFKM3hC3uL7Ioyq7x/vOLeICCN05fb/pH0k= h1:7S9M489iMyHBNxwZnk9/EHS098H4/F6TATF2mIxtB1Q=
 v0.8.2 h1:V4HdnCeaGbFHQUPHJBclcjUtaQkCtXyA3dDNq9WlsLU= -
 v0.8.3 h1:SuHHrKes0cOhSd5ZsPmqqQqCYUYEsMqzl+auGuv9C0Q= -
 v0.9.0 h1:RSQQAbXGArQ0dIDEq+PI6WqN6if+5KHu6x2Cx/GXLTQ= -
github.com/ziutek/mymysql
 v1.4.3 h1:aErmESH5tFoJlyNce76H+xzdKCz3rWTOR4PTGQIFcoE= h1:LMSpPZ6DbqWFxNCHW77HeMg9I646SAhApZ/wKdgO/C0=
 v1.4.4 h1:aVSVBCAGPqR3gUckDZ1W/qK+1UtMkxhzyJPRBXonuuw= -
 v1.4.5 h1:55GbxxhfFEq+HT7ZLsC624KrMdqHjdyFOCyH4hgI9pY= -
 v1.4.6 h1:QLpBKr62dLaA+y5u3DsloyfokSMUdhBrcARwPOnYjaU= -
 v1.4.7 h1:uydWD6PsRks2JqJsUgEv4utZX4YoWRkpls8YbOWglLw= -
 v1.4.8 h1:a4yEDRBRmPyaJ7LbcQWzz+EbGjCTxFlGoEAaJ2DKrOM= -
 v1.5.1 h1:BKYujXJ/dWauorDmll1Ef969gaItF+yt88KVsTeOG9k= -
 v1.5.2 h1:VJ39XrU6PCCxSAzJjABMsCZ+xFNz6BbAmhgeiRwaeu0= -
 v1.5.3 h1:CWmvOapD0QgYi7EQnJmZzdQkvwW4dmWtC0xB++zHqQ8= -
 v1.5.4 h1:GB0qdRGsTwQSBVYuVShFBKaXSnSnYYC2d9knnE1LHFs= -
github.com/zorkian/go-datadog-api
 v2.10.0+incompatible h1:Hyuuz5M/9/H0bYMbhtAYgXPmkZBs7jWXE4h0wPiTAak= h1:PkXwHX9CUQa/FpB9ZwAD45N1uhCW4MT/Wj7m36PbKss=
 v2.11.0+incompatible h1:vey15hqOtFsxHLJZ+pWlm7JcOB6I1eFztbdJ8JnkD04= -
 v2.12.0+incompatible h1:WJpBRnfeRv7NXxOjoW+QzPTPOZFi0DF06drMNi4Uufw= -
 v2.13.0+incompatible h1:+FaPrQvjLyVIxg79q+QmjKSTvJj7MOU1CGPTzG2GzPo= -
 v2.14.0+incompatible h1:nbZXx4ZFpTZpz3JFA9davqIvHeF4HCkt+rYk6QPGh70= -
 v2.15.0+incompatible h1:uesfvHD7kO5iWAb4AcHoRPlU8qIj+tqGZFfc+YNBQic= -
 v2.16.0+incompatible h1:6ErDLNs84H0tfapGzxw5s4tTndaQWpezM8XGWaBD0Fs= -
 v2.17.0+incompatible h1:saadvbg3EfLHIJWr6P+6cY3nv3a9xOp8/Py0bD8Dq/o= -
 v2.18.0+incompatible h1:7JZOVDO8qDaXDKPAzTgiJahU3IoDyzxbLDwoT0U9n0w= -
 v2.19.0+incompatible h1:G17NtfeHwrsQ2degevkwiVEirZENSOjbGOjLLT/kwa4= -
go.uber.org/zap
 v1.3.0 h1:aS854BDNgXguGUPznjhF3oE3OEAFULOOjhsZdTnUNsk= h1:vwi/ZaCAaUcBkycHslxD9B2zi4UTXhF60s6SWpuDF0Q=
 v1.4.0 h1:1NCAUAdWsNejrjT3CHBdCbC20Ztt9AfByBV0NMX6Sao= -
 v1.4.1 h1:SNpwY112Mv6x3CAt0P9fKKXYIec9Ocx34g5+iP/uzas= -
 v1.5.0 h1:0cMK/D2qNl7WS9EJOjIlMhOqqQdp2ol6yZHqS5ny/vw= -
 v1.6.0 h1:8sEbWqRDmX8SVFnlnV2itfOmd0Q6fUXlBcQNbss6RCE= -
 v1.7.0 h1:a/j0E3H61zEkGCIRXd5Ympunuf8BoNDpro1w04oAT1w= -
 v1.7.1 h1:wKPciimwkIgV4Aag/wpSDzvtO5JrfwdHKHO7blTHx7Q= -
 v1.8.0 h1:r6Za1Rii8+EGOYRDLvpooNOF6kP3iyDnkpzbw67gCQ8= -
 v1.9.0 h1:WciNIN5xKjsLnAkg/+UngCXim8pdyhP/4QwycrDzQlQ= -
 v1.9.1 h1:XCJQEf3W6eZaVwhRBof6ImoYGJSITeKWsyeh3HFu/5o= -
goji.io
 v1.0.0 h1:0MrfABbPhQ6s+zi9Qq80zOw5gqJIIX1jrjGiop49erA= h1:sbqFwrtqZACxLBTQcdgVjFh54yGVCvwq8+w49MVMMIk=
 v1.1.0 h1:7QNQWwPiGPyOlpH9UDNh+F9BdGMLXr60lfvPSnigA8o= -
 v2.0.0+incompatible h1:QY6NuzeDeRk+8Iby4IfuN/k0d82K+fDFslQF2I2f6AM= -
 v2.0.2+incompatible h1:uIssv/elbKRLznFUy3Xj4+2Mz/qKhek/9aZQDUMae7c= -
golang.org/x/text
 v0.1.0 h1:LEnmSFmpuy9xPmlp2JeGQQOYbPv3TkQbuGJU3A0HegU= h1:NqM8EUOU14njkJ3fqMW+pc6Ldnwhi/IjpwHt7yyuwOQ=
 v0.2.0 h1:WtDSLEtcB5GqbjSlyn8XcYtxjw+SgFMc2RILOvq7CuE= -
 v0.3.0 h1:g61tztE5qeGQ89tm6NTjjM9VPIm088od1l6aSorWRWg= -
google.golang.org/api
 v0.1.0 h1:K6z2u68e86TPdSdefXdzvXgR1zEMa+459vBSfWYAZkI= h1:UGEZY7KEX120AnNLIHFMKIo4obdJhkp2tPbaPlQx13Y=
google.golang.org/appengine
 v1.0.0 h1:dN4LljjBKVChsv0XCSI+zbyzdqrkEwX5LQFUMRSGqOc= h1:EbEs0AVv82hx2wNQdGPgUI5lhzA/G0D9YwlJXL52JkM=
 v1.1.0 h1:igQkv0AAhEIvTEpD5LIpAfav2eeVO9HBTjvKHVJPRSs= -
 v1.2.0 h1:S0iUepdCWODXRvtE+gcRDd15L+k+k1AiHlMiMjefH24= h1:xpcJRLb0r/rnEns0DIKYYv+WjYCduHsrkT7/EB5XEv4=
 v1.3.0 h1:FBSsiFRMz3LBeXIomRnVzrQwSDj4ibvcRexLG0LZGQk= -
 v1.4.0 h1:/wp5JvzpHIxhs/dumFmF7BXTf3Z+dd4uXta4kVyO508= -
google.golang.org/cloud
 v0.29.0 h1:rRGVXkmZfWZdNdpGVX1wLul3svXC6GJH9WEF1bxHbJo= h1:0H1ncTHf11KCFhTc/+EFRbzSCOZx+VUbRMk55Yv5MYk=
 v0.30.0 h1:ZsPgUufmWaDqeFDnVJex3CAukBTXBQzuju5JmGbr/Yg= -
 v0.31.0 h1:DGMTB5kXUzCbsSIzpyWx+gBxkJjF948r0+8jjEyFAdY= -
 v0.32.0 h1:emDKeqwANbquAQQ5ib/ZEyJOZG7ZVoGcM73blWc4o9I= -
 v0.33.0 h1:TI5c/hhJxDVfa9SmQbJ5AaG3ZLaxyPt4ekeSrBxFtcY= -
 v0.33.1 h1:oGKfwbLysKBV9P5j7vO87xLeQ5v8YZBKl3nTq7qgG4A= -
 v0.34.0 h1:RcDvK+lZ4C9TgF3jLRYPbmr7wVf7h2+Eop65v38SWKQ= -
 v0.35.0 h1:9fXA4nVWICgKfuf5xLPxv2HrqhZUADiPBeozpS6iECo= h1:UE4juzxiHpKLbqrOrwVrKuaZvUtLA9CSnaYO+y53jxA=
 v0.35.1 h1:rOKGx+qJQGpyiSdmIDZ53PU+YS2Qo4BY1yTSCgAh1Lo= h1:wfjPZNvXCBYESy3fIynybskMP48KVPrjSPCnXiK7Prg=
 v0.36.0 h1:+fAUIrmx61hOm16RB316xZrdCb+ibzAlQkaKhbgeB+E= h1:RUoy9p/M4ge0HzT8L+SDZ8jg+Q6fth0CiBuhFJpSV40=
google.golang.org/grpc
 v1.11.3 h1:yy64MFk0j8qZbdXVA0MaSE+s/+6nCUdiyf1uNSjAz0c= h1:yo6s7OP7yaDglbqo1J04qKzAhqBH6lvTonzMVmEdcZw=
 v1.12.0 h1:Mm8atZtkT+P6R43n/dqNDWkPPu5BwRVu/1rJnJCeZH8= -
 v1.12.1 h1:KwYoZQdVL6qeXTAa00oUFpVwXn54zQF6Bw5RGKdJZfs= -
 v1.12.2 h1:FDcj+1t3wSAWho63301gD11L6ysvOl7XPJ0r/ClqNm0= -
 v1.13.0 h1:bHIbVsCwmvbArgCJmLdgOdHFXlKqTOVjbibbS19cXHc= -
 v1.14.0 h1:ArxJuB1NWfPY6r9Gp9gqwplT0Ge7nqv9msgu03lHLmo= -
 v1.15.0 h1:Az/KuahOM4NAidTEuJCv/RonAA7rYsTPkqXVjr+8OOw= h1:0JHn/cJsOMiMfNA9+DeHDlAU7KAAB5GDlYFpa9MZMio=
 v1.16.0 h1:dz5IJGuC2BB7qXR5AyHNwAUBhZscK2xVez7mznh72sY= -
 v1.17.0 h1:TRJYBgMclJvGYn2rIMjj+h9KtMt5r1Ij7ODVRIZkwhk= h1:6QZJwpn2B+Zp71q/5VxRsJ6NXXVCE5NRUHRo+f3cWCs=
 v1.18.0 h1:IZl7mfBGfbhYx2p2rKRtYgDFw6SBz+kclmxYrCksPPA= -
gopkg.in/alecthomas/kingpin.v1
 v1.2.5 h1:adWKhcoHnJMNx5JS81STHGL/PHN9OR0JF8dd5zxotD0= h1:vs0oy7ub8knYaut5kITUTmx/WeE4xRuEeOR34yEAWEA=
 v1.2.6 h1:irP8y6OKhYcvY0fZNJW24FCyfGuy5PrqVuoSuktmtqY= -
 v1.3.0 h1:dOcwoQwS3AJK0MGT2ztWl24OGzc0a/XRAjSMdSRhMO0= -
 v1.3.1 h1:To7zDWiMUHcnjk//o4K8/wgKcDEvIU8MOh1Y6QTaER8= -
 v1.3.2 h1:GvXF1sXhyYdWbouLRrQuMtJUnP8oXns4lLgXTdTZU2U= -
 v1.3.3 h1:1RqPtOxG4sqUZfPoUoWruJOZdAw8DJ1wXRpz1ZMj7FU= -
 v1.3.4 h1:80/ePhY+TzDoIGQOYu4ND2/YqVl3OiQFdpHx5+z2XIo= -
 v1.3.5 h1:9FRYLl899BwNDjiQ/Dd7a0tQOE9qmObfrrdxAAwtgAA= -
 v1.3.6 h1:KgvpV2bWEsmd9HgHaChiRGUMS6l44SEYt2MKQie22rM= -
 v1.3.7 h1:Wu7NdOktFr6uMMaXIkZ1eDz7z6KMbpVoDCrTbYCUtiA= -
gopkg.in/alecthomas/kingpin.v2
 v2.1.9 h1:7+1JAx2MMOemIxjkXbbaFgZDT4qoLbryecW1rRlRFwA= h1:FMv+mEhP44yOT+4EoQTLFTRgOQ1FBLkstjWtayDeSgw=
 v2.1.10 h1:ALcNBHARUALDg20tS9QjwLk1ePLpeRHA9iqAikXtp0k= -
 v2.1.11 h1:XkypDUTQATD111Q6hJPVuyjVynaJV9DW27v01t91IbM= -
 v2.2.0 h1:12WvoFfGYArkTPMkyiaqzPOZIXkE3Msmkm+waOiHE/c= -
 v2.2.1 h1:CNrHXE2HWCsh6qo/qa4VlJc+xKLtEtPhYDdWNLWyKJI= -
 v2.2.2 h1:VBV8OzdyP4EuRQy9lkr5gkIGaGt5FRC0JH/+TmQVfd8= -
 v2.2.3 h1:/L3oK40poPRwke0Ipa6qqf8n+awu60Vl3DMe+3jLDt4= -
 v2.2.4 h1:CC8tJ/xljioKrK6ii3IeWVXU4Tw7VB+LbjZBJaBxN50= -
 v2.2.5 h1:qskSCq465uEvC3oGocwvZNsO3RF3SpLVLumOAhL0bXo= -
 v2.2.6 h1:jMFz6MfLP0/4fUyZle81rXUoxOBFi19VUFKVDOQfozc= -
gopkg.in/bluesuncorp/validator.v5
 v5.0.2 h1:WSpZKVkKjqEZVGvy3HyGIbtKNfDRCFGMZ9js7NTUhKk= h1:ScQmud/GM3iSR85jRE+8BI8E8oFv5oj4qyd5Xaw7hgE=
 v5.5.1 h1:CQJGYfrkmhivpiAjjlRsd0d6d7S6uQu0wwrEaMGhpqU= -
 v5.6.1 h1:jcgu6U1NgGpL3dtHypAtAeyeXuoYdL2u/8pNepFjc7A= -
 v5.7.1 h1:uo+dcug2WYWrciowmsEpdLoDah/EyDlCA/+4Sdn6qJo= -
 v5.8.1 h1:7zi68O6o6JHkGxcdWG5fHKvMAOT3VwNFoD9JbEoq9JA= -
 v5.9.1 h1:XEU2HtMj0Rki3kmHh+uilvENyWgDEaR5LDLtYsjiumM= -
 v5.9.2 h1:MUpsHxMu0up3jcb8nhJ8Ihag3p8I8M5S/fvMqLSm49c= -
 v5.10.1 h1:Vhq9Q92k5VrG+5MFiB58q4MAywVvnPbyWnHK+HHnYrQ= -
 v5.10.2 h1:JuA9iBB5gBt3QoTCE812ZDA1U0wZE/0EBfeqPnFUiMI= -
 v5.10.3 h1:clgxLhQVQIE5krWHyYuqJralvQ9SkkTh3AdZGeQL2D4= -
gopkg.in/cheggaaa/pb.v1
 v1.0.17 h1:jUdPXsaaO29NA7hH5A6vKQbHsudTgI9yUDYY3TkwKHU= h1:V/YB90LKu/1FcN3WVnfiiE5oMCibMjukxqG/qStrOgw=
 v1.0.18 h1:h5Qflf8N54NDtm3lWfBuCD4rslDjkXDoGkEMZCH4R80= -
 v1.0.19 h1:FiMbj8xLGIsj8TLj3O+0GkiydM2OLJhyerwuyNozYug= -
 v1.0.20 h1:kgQVoCjFPiI1fNjdWthabnG1rOAb+/7Z6KeGk2aeZ/w= -
 v1.0.21 h1:vuB0AoAvfgI2z41QOfHeoANHRbfcs6T2l0jOLUnOYkI= -
 v1.0.22 h1:c9uUtBcJbskglPcslP+bFq43Y9mR+Hja6qPRW0bsOJ0= -
 v1.0.24 h1:+chXvORlyxoK9A04q/RTEMg+wxhyPbUW3Q17mDXthsU= -
 v1.0.25 h1:Ev7yu1/f6+d+b3pi5vPdRPc6nNtP1umSfcWiEfRqv6I= -
 v1.0.26 h1:KbH37VyQGNNrLEz+fflXwuLLxnPNoWwUwBF783VJWUg= -
 v1.0.27 h1:kJdccidYzt3CaHD1crCFTS1hxyhSi059NhOFUf03YFo= -
gopkg.in/dancannon/gorethink.v1
 v1.1.1 h1:Ugc3iyikMWkIHmyyPvvkIgBTj9T4Bs+13XO+rRkBkxQ= h1:iqcWmPUmevjldiIysDCQUzcdoDVi9heinXOgm1hkg8Q=
 v1.1.2 h1:P5b5fvspQiQZwom/12kccUVqDji+glv0JsCMQ+547BQ= -
 v1.1.3 h1:DDW/KU9buolgsQ5uNsuDXY0ypGMxtT0tJ5Nmu6Xwoj4= -
 v1.1.4 h1:zH3I7kmh2jzOeNVHWOBUNFw8UHAFc05OgpHGg2O9SWc= -
 v1.2.0 h1:39YMqRInTt1GyxfGsBqEl3YrKWAukRLeDLwOeW5VGkE= -
 v1.3.0 h1:Yn2ERpLuOsP/N1bWd4UWH9T+oYf6HXxYBfNMkyPAo6M= -
 v1.3.1 h1:Mb1tTMIjZMOnD5kawrcByDTiwI8mw6lu4aCHaXO0TQk= -
 v1.3.2 h1:bniPMUXfzXpHxEDJhSPOnZiYHdk9Vi+sM7OnlztZt08= -
 v1.4.0 h1:UV5U7W7wtdvCovPZUhXWCyy/rC9uoOTdYzV2A5x4W6A= -
 v1.4.1 h1:6yZEtwHMaDOBcI2OBAQwpVZoHt9RYg5RcPFnNEHbS9I= -
gopkg.in/errgo.v1
 v1.0.0 h1:n+7XfCyygBFb8sEjg6692xjC6Us50TFRO54+xYUEwjE= h1:CxwszS/Xz1C49Ucd2i6Zil5UToP1EmyrFhKaMVbg1mk=
gopkg.in/fatih/pool.v2
 v2.0.0 h1:xIFeWtxifuQJGk/IEPKsTduEKcKvPmhoiVDGpC40nKg= h1:8xVGeu1/2jr2wm5V9SPuMht2H5AEmf5aFMGSQixtjTY=
gopkg.in/fatih/set.v0
 v0.1.0 h1:aaCY9PUgkH430Tl9sN6N5FqNeEfGgmPnGlY0r9WYZAE= h1:5eLWEndGL4zGGemXWrKuts+wTJR0y+w+auqUJZbmyBg=
 v0.2.0 h1:xGc8mJVI7FVlSeDeURbfEHv/QLTsBYD+pbVE9KAwRmI= -
 v0.2.1 h1:Xvyyp7LXu34P0ROhCyfXkmQCAoOUKb1E2JS9I7SE5CY= -
gopkg.in/fsnotify.v0
 v0.8.10 h1:Rap3u6YAElWRx0HGdGJPY9bSfkG+xgO0fch5elhrdsA= h1:ggSdmL/M3iqOa30tRdm4ctSkKd0e3Gsn8BE1lanSKk8=
 v0.8.11 h1:x+65ylSNqSECOrTu0lo4GBgytbQp4QDEVoNS/ggd6r8= -
 v0.8.12 h1:V5QBxu7lrI4mkdZbbF6cnJr7OyeZENzxWGAGMHt0t9Q= -
 v0.8.13 h1:sGk4jA+Tyy95K5yPiEwvJAQfDHfygsPfxvf1TCQlNMA= -
 v0.9.0 h1:TRsGdVL6CM/HXMefUTFQn9OsCJdEJ+w9reTUqEnTLK4= -
 v0.9.1 h1:m9KEXc/B7HfQ8bprjSWJ323b8lnPW1lwlUxjvGWe7+o= -
 v0.9.2 h1:2yd5kioi95SaVx+WbkTaFnfuYFcwOW+tXnjOBJPCSIY= -
 v0.9.3 h1:EE38OZZkLmA44BsS+DCgO8BjptBMi3IbwTAUuKwU16k= -
gopkg.in/fsnotify.v1
 v1.2.8 h1:BHtBmcj32Obn0a1gQtl294WlsVxsJk8BaiACmPRxHIg= h1:Tz8NjZHkW78fSQdbUxIjBTcgA1z1m8ZHf0WmKUhAMys=
 v1.2.9 h1:sQ4u2nqc93srqGq6uKE2mzSxC2XJF34qBut5caJJUQw= -
 v1.2.10 h1:YUf875DvHA1ayPYGVkHsX7ImtrltD4OpV6C97Lxyk4g= -
 v1.2.11 h1:56IABVxiq+71VfokWUEs8epu6/TyEVWEvzBTmY3B4EE= -
 v1.3.0 h1:JOfhQrGyc8jb766XEc2vcjohQoQW9pGaFr8b1Kj2az0= -
 v1.3.1 h1:MT8ythqog2cHQggwIrobAxM9m6sRB/uT37DrtskBpSg= -
 v1.4.0 h1:Hf3cBkm0mI2PqJw9kjz/T5JF5MeHLfgWpPVTd4GPgP8= -
 v1.4.1 h1:NKzHGradUToAhPGR9Ff1ODW7igWcNHUw/Jaf1aCwJco= -
 v1.4.2 h1:AwZiD/bIUttYJ+n/k1UwlSUsM+VSE6id7UAnSKqQ+Tc= -
 v1.4.7 h1:xOHLXZwVvI9hhs+cLKq5+I5onOuwQLhQwiu63xxlHs4= -
gopkg.in/gcfg.v1
 v1.0.0 h1:7cD+2eVCvV3sE7pinuyCwpvP+BZnaoPrLk9nfQJbzBk= h1:yesOnuUOFQAhST5vPY4nbZsb/huCgGGXlipJsBn0b3o=
 v1.1.0 h1:I1iRlVxpuOHvE5s8Pw1mqGeQ2DJJ7YFUCg+JgoxiGeI= -
 v1.2.0 h1:0HIbH907iBTAntm+88IJV2qmJALDAh8sPekI9Vc1fm0= -
 v1.2.1 h1:wJld/fq1ChPq0K12xrOWpH9E0708XZpQK05DUY0tZmk= -
 v1.2.2 h1:Hzbgbvi2etJFuJvwqC1N+mMGTzojZ9niWXv3dmTEwEA= -
 v1.2.3 h1:m8OOJ4ccYHnx2f4gQwpno8nAX5OGOh7RLaaz0pj3Ogs= -
gopkg.in/go-playground/validator.v8
 v8.0.1 h1:FppjEo03eWfk56wLARtuKd8gxth8YdsOcCIxoYInmR8= h1:RX2a/7Ha8BgOhfk7j780h4/u/RRjR0eouCJSH80/M2Y=
 v8.8.1 h1:M8YWdxbqGCu6nT3HKsYZJb2BILSRmnARwGFika2a1nI= -
 v8.15.1 h1:IrBBTcgklCPw+7tjCGdAErvsk+44VaIeS/4T1utPQ+I= -
 v8.17.1 h1:W1Q1z7rfiJiNhoBkHYqb9TJAdOqPXsyNeZ+8cAzP+kg= -
 v8.17.2 h1:4R86dnE4UskypytrvcWswEtRQ3krktX1mLp7QjCO93o= -
 v8.17.3 h1:4VWLMak6xEPpotvt6qEzKeYxg7sSlEclqtdTmkCiHck= -
 v8.18.0 h1:Vq7TLtmeTm6WYaeH3vtJLuf5Y99IStzJhhcrzZ8TovE= -
 v8.18.1 h1:F8SLY5Vqesjs1nI1EL4qmF1PQZ1sitsmq0rPYXLyfGU= -
 v8.18.2 h1:lFB4DoMU6B626w8ny76MV7VX6W2VHct2GVOI3xgiMrQ= -
gopkg.in/go-playground/validator.v9
 v9.20.1 h1:FgbLL8fWdzeNaZgYGlmEcmXkXZVadKzllL6e1lVHECY= h1:+c9/zcJMFNgbLvly1L1V+PpxWdVbfP1avr/N00E2vyQ=
 v9.20.2 h1:6AVDyt8bk0FDiSYSeWivUfzqEjHyVSCMRkpTr6ZCIgk= -
 v9.21.0 h1:wSDJGBpQBYC1wLpVnGHLmshm2JicoSNdrb38Zj+8yHI= -
 v9.21.1 h1:5TmeijDXOo30RkyYzQKgnm8h04beTpbW66crEqGUHOk= -
 v9.22.0 h1:voA791S2rJ6IWnusR9SL0Z/2e+DrfKlLXaEr+jRHf5Y= -
 v9.23.0 h1:oq297iqu7qsywIbeW5DBUTtV1nV750Y4q+H8MnDh0Yc= -
 v9.24.0 h1:4pXadp8xZVW4WR1Ygw8zDqeCMVHxTGI9tPWyzD2XSzY= -
 v9.25.0 h1:Q3c4LgUofOEtz0wCE18Q2qwDkATLHLBUOmTvqjNCWkM= -
 v9.26.0 h1:2NPPsBpD0ZoxshmLWewQru8rWmbT5JqSzz9D1ZrAjYQ= -
 v9.27.0 h1:wCg/0hk9RzcB0CYw8pYV6FiBYug1on0cpco9YZF8jqA= -
gopkg.in/godo.v2
 v2.0.0 h1:mt97ITWHUj992Vlz+k1qJcwhr/lgOvvRFnkMmSicZeA= h1:wgvPPKLsWN0hPIJ4JyxvFGGbIW3fJMSrXhdvSuZ1z/8=
 v2.0.1 h1:gSerkUE04EIE/2a7A63LXZJ/gMPa8E/HREtS8PZmCT0= -
 v2.0.2 h1:Kph1KeQYhMuLtT99y1Ro4j9hvY+s+lqSamdNvPCM7ww= -
 v2.0.3 h1:T1XOiT9CO9W6VS99qVPtWUD7TJpSoMvXvAAhN2jbdxg= -
 v2.0.4 h1:lOhtfzkbJ45vN4ihMHVlLDXKqQ7y9Z4oU3A6uOO9BPY= -
 v2.0.5 h1:1KIx7P29P4M7XhyO7+i65koUxJ2FbLMZJJnhR/dbthY= -
 v2.0.6 h1:ZQyUuVDn20EyxuH1dCDb9Ue1XTWt+vooVmIbSIBw0tQ= -
 v2.0.7 h1:nUsEBDOFEqd0XhXNsKW4S93VgpsJulT6EwYvD5kCoDI= -
 v2.0.9 h1:jnbznTzXVk0JDKOxN3/LJLDPYJzIl0734y+Z0cEJb4A= -
gopkg.in/gorp.v1
 v1.2.1 h1:t50MTR1csu5McYtgirTr/v/SksNgcZ3+AE24OX3BXZk= h1:Wo3h+DBQZIxATwftsglhdD/62zRFPhGhTiu5jUJmCaw=
 v1.6.1 h1:eDPq0qHmO9DwOsXA8NXDQLKckmtjEWAlpOcUoEI6s2Q= -
 v1.7.1 h1:GBB9KrWRATQZh95HJyVGUZrWwOPswitEYEyqlK8JbAA= -
 v1.7.2 h1:j3DWlAyGVv8whO7AcIWznQ2Yj7yJkn34B8s63GViAAw= -
gopkg.in/guregu/null.v3
 v3.0.1 h1:bPjE2K6nWBZ/FwopjdG4K6DEF02ltwZw5oPT1wOipfg= h1:E4tX2Qe3h7QdL+uZ3a0vqvYwKQsRSQKM5V4YltdgH9Y=
 v3.2.0 h1:qHvBLdOZhlFfEsCxdGDE0cJH2F0TU8YXb5IrFHBpsoU= -
 v3.2.1 h1:XwY4n+l1y+DQXgbkJSgcsg3UMsJTCNrDkNLnXpZ9FpU= -
 v3.3.0 h1:8j3ggqq+NgKt/O7mbFVUFKUMWN+l1AmT5jQmJ6nPh2c= -
 v3.4.0 h1:AOpMtZ85uElRhQjEDsFx21BkXqFPwA7uoJukd4KErIs= -
gopkg.in/igm/sockjs-go.v2
 v2.0.0 h1:NfDyi1jrF9v2VOPESefhKH1NRqpoE9tp4v6kxVR3ubs= h1:xvdpHZ3OpjP0TzQzl+174DglrrnYZKVd6qHPIX20Z1Q=
gopkg.in/inf.v0
 v0.9.0 h1:3zYtXIO92bvsdS3ggAdA8Gb4Azj0YU+TVY1uGYNFA8o= h1:cWUDdTG/fYaXco+Dcufb5Vnc6Gp2YChqWtbxRZE0mXw=
 v0.9.1 h1:73M5CoZyi3ZLMOyDlQh031Cx6N9NDJ2Vvfl76EDAgDc= -
gopkg.in/ini.v1
 v1.38.1 h1:8E3nEICVJ6kxl6aTXYp77xYyObhw7YG9/avdj0r3vME= h1:pNLf8WUiyNEtQjuu5G5vTm06TEv9tsIgeAvK8hOrP4k=
 v1.38.2 h1:dGcbywv4RufeGeiMycPT/plKB5FtmLKLnWKwBiLhUA4= -
 v1.38.3 h1:ourkRZgR6qjJYoec9lYhX4+nuN1tEbV34dQEQ3IRk9U= -
 v1.39.0 h1:Jf2sFGT+sAd7i+4ftUN1Jz90uw8XNH8NXbbOY16taA8= -
 v1.39.1 h1:rLTmwleNumItfQvoOzEoMVnxhExiFsc1AVT/R99UOCA= -
 v1.39.2 h1:TWzeigUv2RIRCp+09pwfJemxPXRn8AI7P4Ow5NYxIYc= -
 v1.39.3 h1:+LGDwGPQXrK1zLmDY5GMdgX7uNvs4iS+9fIRAGaDBbg= -
 v1.40.0 h1:JOoHKRa3vZxx47SL6sOY0gj0hfmA24l+BkQ4CftFizc= -
 v1.41.0 h1:Ka3ViY6gNYSKiVy71zXBEqKplnV35ImDLVG+8uoIklE= -
 v1.42.0 h1:7N3gPTt50s8GuLortA00n8AqRTk75qOP98+mTPpgzRk= -
gopkg.in/ldap.v2
 v2.2.1 h1:uEGgNMi/p26RvB/tgI+Mcp0iU2sk7gActz6JUQlr0zM= h1:oI0cpe/D7HRtBQl8aTg+ZmzFUAvu4lsv3eLXMLGFxWk=
 v2.2.2 h1:5qrQsQnMAMaiw7RiKL5BdJ8voqgj9IAHO4PdhC7uvu8= -
 v2.3.0 h1:vwU7LAjFERkhFo+UiiJ70sMaE6KEgCrB6KTWjP4ZhO0= -
 v2.4.0 h1:vXIxs+LA+6RKbdWYwAUaHZB3WH45V3HhRpNjGJBRXCw= -
 v2.4.1 h1:hpjvxi3mgMuQoUY5tEC4/y+bo8IkS/5auzmZqiAtIeU= -
 v2.5.0 h1:1rO3ojzsHUk+gq4ZYhC4Pg+EzWaaKIV8+DJwExS5/QQ= -
 v2.5.1 h1:wiu0okdNfjlBzg6UWvd1Hn8Y+Ux17/u/4nlk4CQr6tU= -
gopkg.in/macaron.v1
 v1.1.11 h1:PBBsp2LefPe+XIoRMrxR0l4Ef+d7qLcmO0Jk23o3rF8= h1:PrsiawTWAGZs6wFbT5hlr7SQ2Ns9h7cUVtcUu4lQOVo=
 v1.1.12 h1:VNgwUqv5EPXTryXMcnU+kq9eCuojFL3RCjKekMNbbds= -
 v1.2.0 h1:REcIkgUoaYbcfwH/Q7w6ndSAwuhXdRGPpcGUXj4amUs= -
 v1.2.1 h1:V5mrRgXNCxnUDBwydwrDf/Prr6rnT50P0el2MY4o8LY= -
 v1.2.2 h1:KpY9I5T9SyYZaE16kSDVi1hTWRGp9vQL71+2tTdP9ys= -
 v1.2.3 h1:nrePeGPQ/xxQwAtPs9arowkWhjIs0pqUyGTowPJDTHU= -
 v1.2.4 h1:hBOy4AsYKd4lC59YjpDl7MGXEss+JYhH8aF/fXf/Cc4= -
 v1.3.0 h1:WzPx4dAD2r4X83porPurl0kfLZ4pt8ItQnr3Paasjuk= -
 v1.3.1 h1:IdmGJaqXWUdEeN7fmhHF3voSAMzSthjZmTV4SkdSW/s= -
 v1.3.2 h1:AvWIaPmwBUA87/OWzePkoxeaw6YJWDfBt1pDFPBnLf8= -
gopkg.in/natefinch/lumberjack.v2
 v2.0.0 h1:1Lc07Kr7qY4U2YPouBjpCLxpiyxIVoxqXgkXLknAOE8= h1:l0ndWWf7gzL7RNwBG7wST/UCcT4T24xpD6X8LsfU/+k=
gopkg.in/olivere/elastic.v2
 v2.0.52 h1:MZd8vVO0DEHBOmqpC738Tpht4Hk7p95Yr1byR8uTWW0= h1:CTVyl1gckiFw1aLZYxC00g3f9jnHmhoOKcWF7W3c6n4=
 v2.0.53 h1:OCejIubg8xLFxnMGwxlGypMTlvnjPQjEna0Ll90eKx8= -
 v2.0.54 h1:BYI8OA86IwbCj7ITT8XMJPUEgzxaKDX+ZKZDcrVMmtY= -
 v2.0.55 h1:WAhhomC40XjwYRI93VAHSKvU8g2vixqPQBToJIyLScg= -
 v2.0.56 h1:ua1qYYbwPAFVTDGrHJ0aa0G9H4JZhcBUkn9q8L4bJ/0= -
 v2.0.57 h1:zdyq4es731u3oGAJHKW8vt5OIEeIeyseTug4i+urfSc= -
 v2.0.58 h1:MQ0JYVkpm2vFzeb5Iq6qeIJwEpk2SXm7N8So4fFdaAY= -
 v2.0.59 h1:ykPi80ghvr1cLBLD8wM+80Wq7bQJPUFBhn8exaMYNs0= -
 v2.0.60 h1:XHzS/ypiAZAcEKjbKrgTZ93/68izI9cEIiZDij+7xrA= -
 v2.0.61 h1:7cpl3MW8ysa4GYFBXklpo5mspe4NK0rpZTdyZ+QcD4U= -
gopkg.in/olivere/elastic.v3
 v3.0.66 h1:jspimEBZgOP4T7zvChuHIG8rTCObchzXiVYKN9XlUD0= h1:yDEuSnrM51Pc8dM5ov7U8aI/ToR3PG0llA8aRv2qmw0=
 v3.0.67 h1:1IP8Q7sEuOvh+MCIO2diQKUyp/ROCtEgJnylZHrDTiU= -
 v3.0.68 h1:OsczWb4iM2WlB1+iyAbKz07GsMdb0V3COUZJxOXEV4Q= -
 v3.0.69 h1:Sya2Bzgtd9TymDanrKi5xT6Egs27DytfQjGk2l5qK2Y= -
 v3.0.70 h1:h/hscpZxZt8wmx6J9n+U4aMAQ0pcgT2dyrVbo7k19Qw= -
 v3.0.71 h1:HwBpAOj8XGnPUfgdO3ozJiLBYu/cBi6/4RZ+QMMw6zc= -
 v3.0.72 h1:6B82T8fxH7ODdwjrxVrxvHeiVdpXxtMIHAChJDyBfyI= -
 v3.0.73 h1:faw+mH0ObCIFjrZFxLrtpfy9+WkM2/HLRGs/3AqgGgE= -
 v3.0.74 h1:55PbM2ONjZ2lQX4EO1/iJzw2CPPNmmlBtwKxuRBPGFI= -
 v3.0.75 h1:u3B8p1VlHF3yNLVOlhIWFT3F1ICcHfM5V6FFJe6pPSo= -
gopkg.in/olivere/elastic.v5
 v5.0.69 h1:SILXLfQjYYG8BkkhEPHgYW9UARDvYmPNiCTu0ZaBAJE= h1:FylZT6jQWtfHsicejzOm3jIMVPOAksa80i3o+6qtQRk=
 v5.0.70 h1:DqFG2Odzs74JCz6SssgJjd6qpGnsOAzNc7+l5EnvsnE= -
 v5.0.71 h1:HLydL4YNSbE7XokUf7B5XxaznoVrDaRaUZ3fRdhBndI= -
 v5.0.73 h1:JsgYVgCE5yLUVwhO9+YjI10/uKGQfrD1Z6Q8W9fCNsI= h1:vZXWaUsyb5y7tXaKN6wwNr89UXvu9pX2QObsJh43kFo=
 v5.0.74 h1:gYU5Ou+i/SlIg5Y99wS6fVCr7qv6SPUHI39t+WCBWcs= h1:uhHoB4o3bvX5sorxBU29rPcmBQdV2Qfg0FBrx5D6pV0=
 v5.0.75 h1:l2+tYJLseGgWpcQXMpGFcV2T3JUBIYmP+PZE/K3XbvU= -
 v5.0.76 h1:A6W7X4yLPQDINHiYAqIwqev+rD5hIQ4G0e1d5H//VXk= -
 v5.0.77 h1:u5NMYGdddkoWl5+5qWqBmqWgCaYrCftNoTiUkWcqs/w= -
 v5.0.78 h1:SI2vT16LLRqLw+ckMSn3RuFYPFg+I6eUMprH3yz855w= -
 v5.0.79 h1:q+FQfSQxl+xIHoEwq8RGBsb5pRB9f8rfaLh4D9jx18A= -
gopkg.in/redis.v2
 v2.3.1 h1:7rf3dKtDXiNLld8rsy3ePEIBRHV+/qt2oOnRfjTqrHs= h1:4wl9PJ/CqzeHk3LVq1hNLHH8krm3+AXEgut4jVc++LU=
 v2.3.2 h1:GPVIIB/JnL1wvfULefy3qXmPu1nfNu2d0yA09FHgwfs= -
gopkg.in/redis.v3
 v3.5.0 h1:PwukUTFZ+z5z/1susHf3C95Damu/yaFXJOfHM9pGlKI= h1:6XeGv/CrsUFDU9aVbUdNykN7k1zVmoeg83KC9RbQfiU=
 v3.5.1 h1:EdMcdTJBsnOL4Cui4dwgHKfRsVERbdFA/zYna9329mo= -
 v3.5.2 h1:7rJ90fIkMvXJw4feZJ8D6DhaASOe/5fwCSZeFff8itQ= -
 v3.5.3 h1:sks8GNNLYWsSceBYj5flEW/UrFe8En1yovF2+/j0BaI= -
 v3.5.4 h1:4r1j1QRGqRuqeWjEvLy6VzyEsF6otx6rC/yiFSkHkRU= -
 v3.6.0 h1:OF0Dl0A94NUG3EIfIX9SByjsZaKuQ4fZkBTDhbqziVk= -
 v3.6.1 h1:WeAQhMHxdLnS+z/axTJwsilXmF1nmerzU8/BCFqCLWA= -
 v3.6.2 h1:VYqbkiS2m8OFS6YO49USaeklXVXbZPGChqf2i7o21cE= -
 v3.6.3 h1:2+KxEZazco1bqNJ+vsMbs2MnSiSf0ck8AF8HOqDo3OU= -
 v3.6.4 h1:u7XgPH1rWwsdZnR+azldXC6x9qDU2luydOIeU/l52fE= -
gopkg.in/redis.v4
 v4.1.8 h1:jjqbmgEkaissIv7weyr6c0YAdJrVF9EtZaVdwRn312o= h1:8KREHdypkCEojGKQcjMqAODMICIVwZAONWq8RowTITA=
 v4.1.9 h1:HRLu1932aeDU7u0HIWLSqQLYFGwly+sUKYLh8N9WwoY= -
 v4.1.10 h1:rkS6GO9nB8YUriCOKbLgg50qUA51zJu9R+nxnyY1TV4= -
 v4.1.11 h1:nuA8xUx2S4J1qQHIZwXH92fHB9SquYLvvOHl8/5UOpU= -
 v4.1.12 h1:PzZR13P1qKQ9rkQyV0QBOT4/OBmRKED/ZxDOL/vrihI= -
 v4.2.0 h1:mQXuDqI+j+Iq544Yo+PA8GdY81vTpS+Da3antVsSfOs= -
 v4.2.1 h1:OmsloTtfG2PO9RHN5+essTITBjdJyi5paxJZxYSpqIQ= -
 v4.2.2 h1:/VNmyVvFYHWdxn6JrG8gFmIqg7jADwA95aXTLjj1cQM= -
 v4.2.3 h1:0s27sbgN3FDHgaBtOme2JBab/AormYY7wQpFR1z8JMw= -
 v4.2.4 h1:y3XbwQAiHwgNLUng56mgWYK39vsPqo8sT84XTEcxjr0= -
gopkg.in/redis.v5
 v5.2.0 h1:65WtsP4bUKiJRzNoG6rvlBuJdiUBnbQz/1gaSFMb0e0= h1:6gtv0/+A4iM08kdRfocWYB3bLX2tebpNtfKlFT6H4mY=
 v5.2.1 h1:FAJ2+937QQWp5jECvq7X4Qs2Hf04nML5w5Ql44kgz/s= -
 v5.2.2 h1:uDAK+ci8pHLmeSD4qfiPG9jFUWMIZr79PU95h+rVr44= -
 v5.2.3 h1:W27CZnhjz4AxBC4x2x2u+C5GqFcYym2e9JUPQx13ZC4= -
 v5.2.4 h1:XpIcn07AdbjbLZOqa6ipEm9aUhLwpgsm3yi0liZMycc= -
 v5.2.5 h1:5rRADpaSVEga31nAvS34KrgrXGFZPgeQSle5CJ9dPdQ= -
 v5.2.6 h1:IfFCluQuDoA4oT/6RjF90XPGSJnWwgcrJQ3RldDJuxs= -
 v5.2.7 h1:uSF0VCKekjBAxSHRMVGgS8gyRsumywOLeud8f/eSDg0= -
 v5.2.8 h1:CurB5zpl0Mg91TGdRUiRuQNShq4M3kL/oOT95oN3xIM= -
 v5.2.9 h1:MNZYOLPomQzZMfpN3ZtD1uyJ2IDonTTlxYiV/pEApiw= -
gopkg.in/russross/blackfriday.v2
 v2.0.0 h1:+FlnIV8DSQnT7NZ43hcVKcdJdzZoeCmJj4Ql8gq5keA= h1:6sSBNz/GtOm/pJTuh5UmBK2ZHfmnxGbl2NZg1UliSOI=
gopkg.in/square/go-jose.v1
 v1.0.0 h1:nwaqj4pUTyiyO9A2teeBt3m+nPD+gjOhgLEHOZWbUeg= h1:QpYS+a4WhS+DTlyQIi6Ka7MS3SuR9a055rgXNEe6EiA=
 v1.0.1 h1:jVeiLyj8z2C00p+KFfRmVWhqXQEnwoMDLt3fJEodlPY= -
 v1.0.2 h1:K8g79YTjmCjOzz9rWgHs4ZiQkH8XLJIoQBYJ7IDoHfo= -
 v1.0.3 h1:3epMv0BDtfRLoRULAg1TT2WcP/hA3BFJgbJBPnlgTIw= -
 v1.0.4 h1:xh1t8LKyh647lM6QqZGZpwTnN+kkqhcTPqjrDhWTn9Y= -
 v1.0.5 h1:BNd6CxBN6s7x00GjOvsJcz1H7RDAh3Czn8NSBlJSqPg= -
 v1.1.0 h1:T/KcERvxOFKL2QzwvOsP0l5xRvvhTlwcTxw5qad61pQ= -
 v1.1.1 h1:pA7KxQLcwADLRJ3lpUC+vIe4LCO8oRBMoq1HJoJhA3U= -
 v1.1.2 h1:/5jmADZB+RiKtZGr4HxsEFOEfbfsjTKsVnqpThUpE30= -
gopkg.in/square/go-jose.v2
 v2.1.3 h1:/FoFBTvlJN6MTTVCe9plTOG+YydzkjvDGxiSPzIyoDM= h1:M9dMgbHiYLoDGQrXy7OpJDJWiKiU//h+vD76mk0e1AI=
 v2.1.4 h1:2F80z1AzUrkIpMSw+HIa5WJBzPTFEzQiTTPLES3hbfI= -
 v2.1.5 h1:xrxLbb+CJOyjQQaau6hz83i+KLufbDAlIX4Y6sejEsM= -
 v2.1.6 h1:oB3Nsrhs3CNwP1t2WZ/eGtjH8BQhmcGx3zD8Lla+NjA= -
 v2.1.7 h1:4m8fIwX7Xdw2WlFiPJtcVCDX6ELrIdpHnRmE6Uqmktk= -
 v2.1.8 h1:yECBkTX7ypNaRFILw4trAAYXRLvcGxTeHCBKj/fc8gU= -
 v2.1.9 h1:YCFbL5T2gbmC2sMG12s1x2PAlTK5TZNte3hjZEIcCAg= -
 v2.2.0 h1:0kdiskBe/uJirf0T5GGmZlS8bWRYUszavQpx91WycKs= -
 v2.2.1 h1:uRIz/V7RfMsMgGnCp+YybIdstDIz8wc0H283wHQfwic= -
 v2.2.2 h1:orlkJ3myw8CN1nVQHBFfloD+L3egixIa4FvUP6RosSA= -
gopkg.in/src-d/go-git.v4
 v4.4.1 h1:acuY71VVmQUSFZSfcO1V3Gt0kajuvL26Up8ZBdB6CI8= h1:CzbUWqMn4pvmvndg3gnh5iZFmSsbhyhUWdI0IQ60AQo=
 v4.5.0 h1:6VjUh+5ATbfmlCAhV/Fb+1uQ7GnwLIuBPkwcRtxHZkk= -
 v4.6.0 h1:3XrA9Qxiwfj7Iusd7dVYUqxMjJYPsLuBdUeQbwnL/NQ= -
 v4.7.0 h1:WXB+2gCoRhQiAr//IMHpIpoDsTrDgvjDORxt57e8XTA= -
 v4.7.1 h1:phAV/kNULxfYEvyInGdPuq3U2MtPpJdgmtOUF3cghkQ= h1:xrJH/YX8uSWewT6evfocf8qsivF18JgCN7/IMitOptY=
 v4.8.0 h1:dDEbgvfNG9vUDM54uhCYPExiGa8uYgXpQ/MR8YvxcAM= h1:Vtut8izDyrM8BUVQnzJ+YvmNcem2J89EmfZYCkLokZk=
 v4.8.1 h1:aAyBmkdE1QUUEHcP4YFCGKmsMQRAuRmUcPEQR7lOAa0= -
 v4.9.0 h1:Khe+oTSklf4aZ1037ayWhx1bwolheNTG6mj0Ss1zrfQ= -
 v4.9.1 h1:0oKHJZY8tM7B71378cfTg2c5jmWyNlXvestTT6WfY+4= -
 v4.10.0 h1:NWjTJTQnk8UpIGlssuefyDZ6JruEjo5s88vm88uASbw= -
gopkg.in/telegram-bot-api.v4
 v4.2.0 h1:IHqSfGmsospUSY9Ln07udR7ru2/swRaaPRnYqKsCRig= h1:5DpGO5dbumb40px+dXcwCpcjmeHNYLpk0bp3XRNvWDM=
 v4.2.1 h1:TDknppPv/X8IcXalSEB8+lUGICaR5qrtgxE4azOUsEU= -
 v4.3.0 h1:F4b2bbhN0K3q69WOj61X55z8JA4Lnce7fnOdk2zn++E= -
 v4.4.0 h1:6a24pbShYp5rbKf6kS2JWa9+0YPm8OxcBbqhL28fWd0= -
 v4.5.0 h1:8jZJyG3auCfkAPcAVr5jZ+ecL7ryPxIXgG0eWyJ8GME= -
 v4.5.1 h1:ESWxojO4ZfhAkZwKhqUN+y/ZMq247jKjlcQAH3YTTTw= -
 v4.6.1 h1:dv01Nt/N1XQHtKYXCw0IvwLZ61G+dTa9sradBjyAqU4= -
 v4.6.2 h1:oUu8dyT/KzUWusD3OiPWTLoFM6n9uskQlW1GLoxjQcE= -
 v4.6.3 h1:f4hJ3ITtvfOj9jr1i1/7GqqMEO9vpfJqbXIsgMUvIs8= -
 v4.6.4 h1:hpHWhzn4jTCsAJZZ2loNKfy2QWyPDRJVl3aTFXeMW8g= -
gopkg.in/tylerb/graceful.v1
 v1.2.6 h1:ckW3yVSRHtWGFJJUSlAv6kG8KIyW6M2P7Vlv9HSVV1A= h1:yBhekWvR20ACXVObSSdD3u6S9DeSylanL2PAbAC/uJ8=
 v1.2.7 h1:qO24UH8UijdTFYyGfno4FylsUqceXtdD7qmZYz/q4kk= -
 v1.2.8 h1:NCopaPkEh50x1Zksp2cyh6wcJ4cjpihHKwn838r6F+8= -
 v1.2.9 h1:JEqK2MKfstLJ/Ck1xRU3ZbyPxXtlyMb659qx3SD1NyA= -
 v1.2.10 h1:b2KKG/98fpUz8cTAyk9VsirN/DB/aG4t3YYMiP9xUBE= -
 v1.2.11 h1:gG4yKasbCr2M5PmKDcSD8QvAAEBw3bLzvA+ZSnzndTk= -
 v1.2.12 h1:fyiwgP14Pt2zcG52YE7jGpmI+Z19EPBr0iPTl1LwJzg= -
 v1.2.13 h1:UWJlWJHZepntB0PJ9RTgW3X+zVLjfmWbx/V1X/V/XoA= -
 v1.2.14 h1:oiz4kQ+TpHevusS0Rj4bHh9P9wL9RhiNaGII/C9W0QU= -
 v1.2.15 h1:1JmOyhKqAyX3BgTXMI84LwT6FOJ4tP2N9e2kwTCM0nQ= -
gopkg.in/unrolled/render.v1
 v1.0.0 h1:f/6S5YVJqWYcqYtvfsv+Eb+A1CWhuA7n+ILks5qv9gw= h1:D8ZfMFuggVdNUNlNz/R8zVjPPHGyMxLuJPA+MSx8na0=
gopkg.in/urfave/cli.v1
 v1.15.0 h1:rI0RRRs5LRJpLe6EkPUCrcACRDwseawCxh3XscivV80= h1:vuBzUtMdQeixQj8LVd+/98pzhxNGQoyuPBlsXHOQNO0=
 v1.16.0 h1:/F2aUks+U8n2RyLUx+2TdxAcnT90mdhawpW/EAnt0e0= -
 v1.16.1 h1:o+4cPmtQWNZpx++xZl7DnR8CBVUEXGKPH9vsNmGO9aQ= -
 v1.17.0 h1:PuvzS+NtlnLBSZ6tKY6us1aeLkCZGlpg1bEIsSmIqxk= -
 v1.17.1 h1:hVHy931pfJxrhG/3zYuTCdzkPM2yjGT72OleI/pL8fk= -
 v1.18.0 h1:8UwKRHJhJXVZiD1knmUd7r2GxeJOkeHW7YEI1u0xIDk= -
 v1.18.1 h1:Z65+UJjxGDZHRdsnYU+Q+KCFrjSwc/VVB0X6jvGBd2U= -
 v1.19.0 h1:mf+2mvYi0FaeObyhdxSudZaT2SbDKTI4oJFTfciGzsI= -
 v1.19.1 h1:pkwzWQSFerxgLtkdWlnjwOS+Vd7VCp/Kwdn3kmeflXQ= -
 v1.20.0 h1:NdAVW6RYxDif9DhDHaAortIu956m2c0v+09AZBPTbE0= -
gopkg.in/vmihailenco/msgpack.v2
 v2.7.3 h1:2R71q9j9HMo3qg3fqSPBq7oyVuSq6n+bWoV+OgtFnms= h1:/3Dn1Npt9+MYyLpYYXjInO/5jvMLamn+AEGwNEOatn8=
 v2.7.4 h1:wAy0oc6rFtax+qrPl+UrOyDw9yrTIGFi3Q9cicDZ9l0= -
 v2.8.0 h1:LB/RzJrvbZ78O0ETDVMZo+WcJmYVT51U/iJ1YPeTTvg= -
 v2.8.1 h1:vwNyqK+a1cxU2pyWau1ohPbG9WtLDPXnh6EGsF3K5Ns= -
 v2.8.2 h1:dSHRUEFGQtymjUv7dFZ8nGVPamzUSW79Zr+7JHNs8ps= -
 v2.8.3 h1:jMY6Wf1VPCyOvzD8dcnmL/HCkC7y60+0x6SQE7qJvvc= -
 v2.8.4 h1:z5FtZ+GB3FcM/lZtyIEZlhaUkfIfOZFN9bHOu6qB2RA= -
 v2.8.5 h1:wT+qv7QaQWrm1Li6CbRzw+zqHsNZzy9V/cNOMXwdU4I= -
 v2.9.0 h1:tmB/gene3e/xkXSpFk5xCNwDAtwcDUqMs9oDDudyEF4= -
 v2.9.1 h1:kb0VV7NuIojvRfzwslQeP3yArBqJHW9tOl4t38VS1jM= -
gopkg.in/yaml.v2
 v2.0.0 h1:uUkhRGrsEyx/laRdeS6YIQKIys8pg+lRSRdVMTYjivs= h1:JAlM8MvJe8wmxCU4Bli9HhUf9+ttbYbLASfIpnQbh74=
 v2.1.0 h1:o2qA7KtU0UqgG4G8I4Hw5iXppmCcmuISiLweMFO22M0= -
 v2.1.1 h1:fxK3tv8mQPVEgxu/S2LJ040LyqiajHt+syP0CdDS/Sc= h1:hI93XBmqTisBFMUTm0b8Fm+jr3Dg1NNxqwp+5A1VGuI=
 v2.2.0 h1:ucE2Go3MGv/WipgucyA7X3+4pRLSbl5sd8WaEs60obQ= -
 v2.2.1 h1:mUhvW9EsL+naU5Q3cakzfE91YhliOondGd6ZrsDBHQE= -
 v2.2.2 h1:ZCJp+EgiOT7lHqUV2J862kp8Qj64Jo6az82+3Td9dZw= -
k8s.io/client-go
 v4.0.0-beta.0+incompatible h1:P+YnxLM0z74ESopTIZpD7gCg1A9wgE++vMIpa0v8JDI= h1:7vJpHMYJwNQCWgzmNV+VYUl1zCObLyodBc8nIyt8L5s=
 v4.0.0+incompatible h1:G0T/bjcZWkHyAv3FRcXwTnaEV2IhCwuYhDx/m3JuxUY= -
 v5.0.0+incompatible h1:GRNZwjeW9HjYCAY0+EwNf6Tvp+mXx/uBVVcT8ixJz+s= -
 v5.0.1+incompatible h1:IPZ0cnux5ui8+X8r1HdeFPXucpQ4HyJQigjo1clq1QM= -
 v6.0.0+incompatible h1:QVR0YsL5jUAs8IB2sHb7IANUK6FYv6CpNLpSPke7R2Q= -
 v7.0.0+incompatible h1:kiH+Y6hn+pc78QS/mtBfMJAMIIaWevHi++JvOGEEQp4= -
 v8.0.0+incompatible h1:tTI4hRmb1DRMl4fG6Vclfdi6nTM82oIrTT7HfitmxC4= -
 v9.0.0-invalid+incompatible h1:PAQ787PYfegRAYGYH3E5bgZS7plC4i8Z2Og1iLCCgV4= -
 v9.0.0+incompatible h1:2kqW3X2xQ9SbFvWZjGEHBLlWc1LG9JIJNXWkuqwdZ3A= -
 v10.0.0+incompatible h1:F1IqCqw7oMBzDkqlcBymRq1450wD0eNqLE9jzUrIi34= -
k8s.io/heapster
 v1.5.0-beta.0 h1:QLmWHYNeI0Y2SH2+ywX9pvrl17Krp9aSsZLVkKn8rC4= h1:h1uhptVXMwC8xtZBYsPXKVi8fpdlYkTs6k949KozGrM=
 v1.5.0-beta.1 h1:eYRpxHJihz1PX8rr6cwKdETr+st5DI4kjfVUpJosppY= -
 v1.5.0-beta.2 h1:JMdMJY62NiCvpTvulKT5WxXmZ/ZAsEFA4B2pgRZYE6g= -
 v1.5.0-beta.3 h1:LQSpxHpKWeGoVsRBP0N26tXQPPzGEFTaPca3ed44vLQ= -
 v1.5.0 h1:Tunw4N48Ys4seXluU9Nle0XBZdCPKYuli115MRwdnic= -
 v1.5.1 h1:gQNVK6c7mB7nhfrAIr2qZcnELENq7WxYJFJEpYQYrXc= -
 v1.5.2 h1:P61v06rdxOdUm8s+31D0oheeCPCH2upMWweCUulzSkg= -
 v1.5.3 h1:JGhFcTE91wr2HQ8jsUIMiHUhIjKBRgU91G+bgtnzOc4= -
 v1.5.4 h1:lH2GCZdqRmUKDoyqRgiXbRmIcevaPYTvkguOuYUl8gQ= -
 v1.6.0-beta.1 h1:+ibnpUBhNjN80vvze2ok3Mza0s7nMa1NVZlKtDPvGMQ= -
`
