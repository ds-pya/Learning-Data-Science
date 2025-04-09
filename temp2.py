package com.yourpkg.log

import android.graphics.Color
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.appcompat.widget.AppCompatTextView
import androidx.recyclerview.widget.RecyclerView

class CustomLogAdapter(
    private val onItemClick: (LogLine) -> Unit
) : RecyclerView.Adapter<CustomLogAdapter.LogViewHolder>() {

    private val logList = mutableListOf<LogLine>()

    inner class LogViewHolder(val view: View) : RecyclerView.ViewHolder(view)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): LogViewHolder {
        val textView = AppCompatTextView(parent.context).apply {
            setPadding(16, 8, 16, 8)
            setTextColor(Color.GREEN)
        }
        return LogViewHolder(textView)
    }

    override fun onBindViewHolder(holder: LogViewHolder, position: Int) {
        val log = logList[position]
        (holder.itemView as? TextView)?.apply {
            text = log.message
            setOnClickListener { onItemClick(log) }
        }
    }

    override fun getItemCount(): Int = logList.size

    fun addLog(log: LogLine) {
        logList.add(log)
        notifyItemInserted(logList.size - 1)
    }
}