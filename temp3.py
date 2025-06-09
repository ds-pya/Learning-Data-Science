private fun showInputDialog() {
    val context = requireContext()
    binding = DialogInputBinding.inflate(layoutInflater)

    val sourceNames = listOf("Source A", "Source B", "Source C")
    val checkBoxMap = mutableMapOf<String, CheckBox>()

    // 동적으로 CheckBox 생성
    sourceNames.forEach { name ->
        val checkBox = CheckBox(context).apply { text = name }
        binding.containerSources.addView(checkBox)
        checkBoxMap[name] = checkBox
    }

    // Spinner 설정
    val sourceOptions = listOf("Option 1", "Option 2", "Option 3")
    val spinnerAdapter = ArrayAdapter(context, android.R.layout.simple_spinner_dropdown_item, sourceOptions)
    binding.spinnerSource.adapter = spinnerAdapter

    // Timestamp 클릭 시 Date + Time Picker
    var selectedTimestamp: Long? = null
    binding.editTimestamp.setOnClickListener {
        val calendar = Calendar.getInstance()
        DatePickerDialog(context, { _, year, month, day ->
            TimePickerDialog(context, { _, hour, minute ->
                calendar.set(year, month, day, hour, minute)
                selectedTimestamp = calendar.timeInMillis
                val formatted = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()).format(Date(selectedTimestamp!!))
                binding.editTimestamp.setText(formatted)
            }, calendar.get(Calendar.HOUR_OF_DAY), calendar.get(Calendar.MINUTE), true).show()
        }, calendar.get(Calendar.YEAR), calendar.get(Calendar.MONTH), calendar.get(Calendar.DAY_OF_MONTH)).show()
    }

    // AlertDialog 빌드
    AlertDialog.Builder(context)
        .setTitle("Input")
        .setView(binding.root)
        .setPositiveButton("확인") { _, _ ->
            val selectedSources = checkBoxMap.filter { it.value.isChecked }.keys.toList()
            val selectedSource = binding.spinnerSource.selectedItem.toString()
            val title = binding.editTitle.text.toString()
            val timestamp = selectedTimestamp

            processInput(selectedSources, selectedSource, title, timestamp)
        }
        .setNegativeButton("취소", null)
        .show()
}