<LinearLayout
    android:id="@+id/container_sources"
    android:orientation="vertical"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:layout_marginBottom="16dp"/>

private fun showInputDialog() {
    val context = requireContext()
    val dialogView = LayoutInflater.from(context).inflate(R.layout.dialog_input, null)

    val container = dialogView.findViewById<LinearLayout>(R.id.container_sources)
    val sourceNames = listOf("Source A", "Source B", "Source C", "Source D")

    // 동적으로 체크박스 생성
    val checkBoxMap = mutableMapOf<String, CheckBox>()
    for (name in sourceNames) {
        val checkBox = CheckBox(context).apply {
            text = name
        }
        checkBoxMap[name] = checkBox
        container.addView(checkBox)
    }

    val spinner = dialogView.findViewById<Spinner>(R.id.spinner_source)
    val editTitle = dialogView.findViewById<EditText>(R.id.edit_title)
    val editTimestamp = dialogView.findViewById<EditText>(R.id.edit_timestamp)

    // Spinner 설정
    val options = listOf("Option 1", "Option 2", "Option 3")
    spinner.adapter = ArrayAdapter(context, android.R.layout.simple_spinner_dropdown_item, options)

    // Timestamp 설정 생략 (이전 코드 참고)

    AlertDialog.Builder(context)
        .setTitle("Input")
        .setView(dialogView)
        .setPositiveButton("OK") { _, _ ->
            val selectedSources = checkBoxMap.filter { it.value.isChecked }.keys.toList()
            val selectedSource = spinner.selectedItem.toString()
            val title = editTitle.text.toString()
            val timestamp = editTimestamp.text.toString()

            processInput(selectedSources, selectedSource, title, timestamp)
        }
        .setNegativeButton("Cancel", null)
        .show()
}