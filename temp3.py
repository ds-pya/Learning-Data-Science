source,topic,dist_type,parameter,comment,fallback
app,game,lognorm,"{'median': 120, 'p95': 600}","게임앱 heavy-user가 하루 수시간 → heavy tail → 로그정규",False
app,movie,lognorm,"{'median': 60, 'p95': 300}","OTT/영화 앱, 주 1~2편 소비, binge 시청자 존재",False
app,beauty,lognorm,"{'median': 45, 'p95': 200}","뷰티앱 짧게 자주, 일부 heavy user 존재",False
app,shopping,lognorm,"{'median': 30, 'p95': 150}","세일 시즌 spike → lognormal",False
app,stock,lognorm,"{'median': 20, 'p95': 100}","짧게 자주, 비교적 낮은 median",False
app,real estate,lognorm,"{'median': 15, 'p95': 80}","관심 시기 집중, 낮은 median, high var",False
app,politics,lognorm,"{'median': 20, 'p95': 120}","정치/뉴스앱, 짧게 보는 다수 vs 장시간 소수",False
app,food,lognorm,"{'median': 30, 'p95': 120}","배달앱 점심/저녁 집중, 꼬리 있음",False
app,auto and vehicles,lognorm,"{'median': 25, 'p95': 100}","차량앱 niche, 특정 user heavy-tail",False
app,soccer,lognorm,"{'median': 25, 'p95': 120}","경기 일정 확인, 뉴스 소비",False
app,travel,lognorm,"{'median': 35, 'p95': 160}","여행 준비시 급증, 평소 sparse",False
you,game,lognorm,"{'median': 90, 'p95': 400}","게임 방송/스트리밍, 일부 binge",False
you,celebrity,lognorm,"{'median': 70, 'p95': 300}","연예/아이돌 영상 소비 집중",False
you,beauty,lognorm,"{'median': 60, 'p95': 250}","뷰티 유튜브 영상(튜토리얼/리뷰) → lognormal",False
you,soccer,lognorm,"{'median': 40, 'p95': 200}","경기 하이라이트/분석 영상",False
you,baseball,lognorm,"{'median': 35, 'p95': 150}","야구 경기/뉴스 하이라이트",False
you,travel,lognorm,"{'median': 50, 'p95': 250}","여행 브이로그, 가이드 영상",False
you,food,lognorm,"{'median': 45, 'p95': 180}","먹방/레시피 영상 소비",False
you,stock,lognorm,"{'median': 25, 'p95': 120}","경제/주식 유튜브 채널 소비",False
web,stock,lognorm,"{'median': 25, 'p95': 100}","증권 뉴스/차트 → 짧고 빈번, 일부 deep dive",False
web,politics,lognorm,"{'median': 25, 'p95': 100}","정치 기사/토론 → median 낮음",False
web,real estate,lognorm,"{'median': 20, 'p95': 80}","부동산 정보 검색 → 특정 시점 집중",False
web,soccer,lognorm,"{'median': 20, 'p95': 80}","스포츠 기사 소비",False
web,shopping,lognorm,"{'median': 15, 'p95': 60}","온라인 쇼핑 브라우징 → 짧은 세션",False
web,food,lognorm,"{'median': 15, 'p95': 60}","요리법 검색, 맛집 리뷰",False
web,beauty,lognorm,"{'median': 20, 'p95': 80}","뷰티 블로그/커뮤니티 검색",False
web,travel,lognorm,"{'median': 18, 'p95': 75}","여행 블로그, 후기 검색",False
ex,running,poisson,"{'lambda': 3}","러닝: 평균 주 3회 → Poisson",False
ex,soccer,poisson,"{'lambda': 1}","아마추어 경기, 주 평균 1회 이하",False
ex,baseball,poisson,"{'lambda': 1}","야구 동호회 경기 빈도 낮음",False
ex,basketball,poisson,"{'lambda': 1}","농구 동호회 활동 주 1회 수준",False
ex,golf,neg_binom,"{'mean': 0.5, 'var': 1.0}","대부분 0, 일부는 주 1~2회 → 과산포",False
ex,cycle,poisson,"{'lambda': 0.8}","자전거 운동은 주 1회 미만",False
ex,tennis,poisson,"{'lambda': 0.6}","주 0~1회, sparse",False
ex,volleyball,poisson,"{'lambda': 0.5}","주 0~1회, sparse",False
cal,travel,zip,"{'lambda': 0.2, 'pi0': 0.7}","캘린더 여행 이벤트, 대부분 0, 가끔 입력",False
cal,movie,zip,"{'lambda': 0.5, 'pi0': 0.5}","캘린더 영화/공연 등록, 희소",False
cal,celebrity,zip,"{'lambda': 0.3, 'pi0': 0.6}","캘린더 연예 이벤트, 희소",False
cal,soccer,zip,"{'lambda': 0.2, 'pi0': 0.7}","캘린더 스포츠 이벤트, 드묾",False
poi,travel,zinb,"{'mean': 0.5, 'var': 2.0, 'pi0': 0.7}","여행 사진, 대부분 0, 여행 시 폭발적",False
poi,shopping,zinb,"{'mean': 0.7, 'var': 1.5, 'pi0': 0.5}","쇼핑몰 방문 사진, 일부 사용자 집중",False
poi,camping,zinb,"{'mean': 0.3, 'var': 1.0, 'pi0': 0.6}","캠핑 사진, 희소 이벤트",False
noti,movie,zip,"{'lambda': 2.0, 'pi0': 0.2}","영화 앱 알림, 빈도 중간",False
noti,beauty,zip,"{'lambda': 2.0, 'pi0': 0.3}","뷰티 앱 알림, 빈도 중간",False
noti,soccer,zip,"{'lambda': 5.0, 'pi0': 0.1}","스포츠 알림, 경기일정 push → 잦음",False
noti,baseball,zip,"{'lambda': 4.0, 'pi0': 0.1}","야구 경기 알림, 경기일정 push → 잦음",False
fallback_duration,non-cover,lognorm,"{'median': 15, 'p95': 60}","duration 소스의 non-cover 토픽 fallback",True
fallback_count,non-cover,zip,"{'lambda': 0.2, 'pi0': 0.8}","count 소스의 non-cover 토픽 fallback",True