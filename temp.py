package com.yourpkg.settings

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import androidx.fragment.app.Fragment
import androidx.preference.PreferenceFragmentCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.yourpkg.log.CustomLogAdapter
import com.yourpkg.log.LogLine

class CustomFragment : PreferenceFragmentCompat() {

    private var logThread: Thread? = null
    private val tagFilter = "MyAppTag" // logcat 필터링용 태그

    override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
        setPreferencesFromResource(R.xml.main)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View? {
        val prefView = super.onCreateView(inflater, container, savedInstanceState)

        val rootLayout = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        prefView?.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
        )

        val logAdapter = CustomLogAdapter { logLine ->
            logLine.destination?.let {
                navigateToFragment(it)
            }
        }

        val recyclerView = RecyclerView(requireContext()).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
            layoutManager = LinearLayoutManager(requireContext())
            adapter = logAdapter
        }

        rootLayout.addView(prefView)
        rootLayout.addView(recyclerView)

        startLogcatReader(logAdapter, recyclerView)

        return rootLayout
    }

    override fun onDestroyView() {
        super.onDestroyView()
        logThread?.interrupt()
        logThread = null
    }

    private fun startLogcatReader(adapter: CustomLogAdapter, recyclerView: RecyclerView) {
        logThread = Thread {
            try {
                val process = ProcessBuilder("logcat", "-s", tagFilter).start()
                val reader = process.inputStream.bufferedReader()

                while (!Thread.currentThread().isInterrupted) {
                    val line = reader.readLine() ?: break
                    val logLine = LogLine(message = line)  // destination은 나중에 분석해서 지정해도 됨

                    activity?.runOnUiThread {
                        adapter.addLog(logLine)
                        recyclerView.scrollToPosition(adapter.itemCount - 1)
                    }
                }

                reader.close()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        logThread?.start()
    }

    private fun navigateToFragment(fragmentClass: Class<out Fragment>) {
        parentFragmentManager.beginTransaction()
            .replace(requireParentFragment().id, fragmentClass.newInstance())
            .addToBackStack(null)
            .commit()
    }
}