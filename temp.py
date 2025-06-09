override fun onPreferenceTreeClick(preference: Preference): Boolean {
    when (preference.key) {
        "your_preference_key" -> {
            showInputDialog()
            return true
        }
    }
    return super.onPreferenceTreeClick(preference)
}

private fun showInputDialog() {
    val context = requireContext()
    val dialogView = LayoutInflater.from(context).inflate(R.layout.dialog_input, null)

    val checkA = dialogView.findViewById<CheckBox>(R.id.check_source_a)
    val checkB = dialogView.findViewById<CheckBox>(R.id.check_source_b)
    val checkC = dialogView.findViewById<CheckBox>(R.id.check_source_c)

    val spinner = dialogView.findViewById<Spinner>(R.id.spinner_source)
    val editTitle = dialogView.findViewById<EditText>(R.id.edit_title)
    val editTimestamp = dialogView.findViewById<EditText>(R.id.edit_timestamp)

    // Spinner setup
    val options = listOf("Option 1", "Option 2", "Option 3")
    spinner.adapter = ArrayAdapter(context, android.R.layout.simple_spinner_dropdown_item, options)

    // Date & Time Picker on timestamp field
    var selectedTimestamp: Long? = null
    editTimestamp.setOnClickListener {
        val calendar = Calendar.getInstance()
        DatePickerDialog(context, { _, year, month, day ->
            TimePickerDialog(context, { _, hour, minute ->
                calendar.set(year, month, day, hour, minute)
                selectedTimestamp = calendar.timeInMillis
                val formatted = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()).format(Date(selectedTimestamp!!))
                editTimestamp.setText(formatted)
            }, calendar.get(Calendar.HOUR_OF_DAY), calendar.get(Calendar.MINUTE), true).show()
        }, calendar.get(Calendar.YEAR), calendar.get(Calendar.MONTH), calendar.get(Calendar.DAY_OF_MONTH)).show()
    }

    AlertDialog.Builder(context)
        .setTitle("Input")
        .setView(dialogView)
        .setPositiveButton("OK") { _, _ ->
            val sources = mutableListOf<String>()
            if (checkA.isChecked) sources.add("Source A")
            if (checkB.isChecked) sources.add("Source B")
            if (checkC.isChecked) sources.add("Source C")

            val selectedSource = spinner.selectedItem.toString()
            val title = editTitle.text.toString()
            val timestamp = selectedTimestamp

            // 처리
            processInput(sources, selectedSource, title, timestamp)
        }
        .setNegativeButton("Cancel", null)
        .show()
}

private fun processInput(
    sources: List<String>,
    source: String,
    title: String,
    timestamp: Long?
) {
    Log.d("Input", "Sources=$sources, Source=$source, Title=$title, Timestamp=$timestamp")
    // 여기에 실제 처리 로직 작성
}