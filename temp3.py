package com.yourpkg.log

import androidx.fragment.app.Fragment

data class LogLine(
    val message: String,
    val destination: Class<out Fragment>? = null
)