개인용 단말기에서 PoC(Proof of Concept) 테스트를 목적으로 한다면, 구글 플레이 스토어의 복잡한 심사 정책(Service Type 증명 등)을 신경 쓰지 않고 가장 강력하고 확실하게 서비스가 죽지 않도록 설정하는 것이 핵심입니다.
2025년 기준 안드로이드 14~16 버전이 탑재된 개인 기기에서 접근성 서비스를 포그라운드로 고정하는 최적의 방법입니다.
1. 매니페스트 (가장 포괄적인 권한 설정)
개인용 테스트 앱이므로 시스템 권한을 넉넉하게 부여합니다. specialUse 타입은 심사가 까다롭지만, 개인 폰에서는 가장 제약이 적은 타입입니다.
xml
<!-- 필수 권한 -->
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_SPECIAL_USE" />
<!-- 포그라운드 유지력을 높이기 위한 추가 권한 -->
<uses-permission android:name="android.permission.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />

<service
    android:name=".MyAccessibilityService"
    android:permission="android.permission.BIND_ACCESSIBILITY_SERVICE"
    android:foregroundServiceType="specialUse"
    android:exported="true">
    <intent-filter>
        <action android:name="android.view.accessibility.AccessibilityService" />
    </intent-filter>
    <meta-data
        android:name="android.accessibilityservice"
        android:resource="@xml/accessibility_service_config" />
</service>
코드를 사용할 때는 주의가 필요합니다.

2. 접근성 서비스 내부 구현 (PoC용)
알림 채널 생성과 포그라운드 전환을 한 번에 처리합니다.
kotlin
class MyAccessibilityService : AccessibilityService() {

    override fun onServiceConnected() {
        super.onServiceConnected()
        startMyForeground()
    }

    private fun startMyForeground() {
        val channelId = "poc_service_channel"
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        // 1. 알림 채널 생성
        val channel = NotificationChannel(channelId, "PoC Test Service", NotificationManager.IMPORTANCE_LOW)
        notificationManager.createNotificationChannel(channel)

        // 2. 알림 빌드
        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("접근성 서비스 작동 중")
            .setContentText("PoC 테스트를 위해 포그라운드 유지 중")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setOngoing(true) // 사용자가 알림을 지우지 못하도록 설정
            .build()

        // 3. 포그라운드 시작 (Android 14+ 대응)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(1001, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(1001, notification)
        }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // 테스트 로직 작성
    }

    override fun onInterrupt() {}
}
코드를 사용할 때는 주의가 필요합니다.

3. 테스트 시 필수 수동 설정 (개인 폰 전용)
코드 구현보다 중요한 것이 안드로이드 시스템 설정입니다. PoC 앱이 시스템에 의해 죽지 않도록 다음 설정을 수동으로 진행하세요.
배터리 최적화 제외:
앱 정보 -> 배터리 -> '제한 없음(Unrestricted)'으로 설정.
코드에서 요청하려면: startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:$packageName")))
알림 권한 허용:
Android 13 이상에서는 알림 권한이 꺼져 있으면 포그라운드 서비스가 실행되지 않을 수 있으니 반드시 허용하세요.
최근 앱 화면에서 잠금:
최근 사용한 앱 목록에서 해당 앱 카드를 길게 누르거나 메뉴를 눌러 '잠금(Lock this app)'을 설정하면 메모리 정리 시 최우선 순위로 보호됩니다.
개발자 옵션 설정:
개인용 기기라면 개발자 옵션 -> 실행 중인 서비스에서 내 앱이 정상적으로 포그라운드 알림과 함께 실행 중인지 실시간으로 확인할 수 있습니다.
4. PoC 팁: 죽었을 때 자동 재시작
접근성 서비스는 설정에서 켜져 있는 한 시스템이 가급적 살려두려 하지만, 가혹한 환경에서는 죽을 수 있습니다. 이럴 때 Activity나 BroadcastReceiver에서 서비스 상태를 체크하여 다시 깨워주는 보조 로직을 넣으면 더 완벽한 PoC가 됩니다.
