#include "utils/TimerWrapper.hpp"

namespace timer
{

/**
 * @brief 生成并返回一个包含计时信息的表格字符串表示
 *
 * 此函数创建一个表格，格式化并填充计时信息，然后返回表格的字符串表示。
 * 表格中包含了被包装函数的名称、每个函数的执行时间，以及一些格式化设置，如颜色和对齐方式。
 *
 * @return std::string 包含计时信息的表格的字符串表示
 */
void TimerWrapper::TimerShow(std::ostream &stream) const
{
    tabulate::Table table;
    table.format().border_color(tabulate::Color::magenta).locale("zh_CN.UTF-8");

    table.add_row({wrapper_name_, ""});
    table[0].format().font_align(tabulate::FontAlign::center).font_color(tabulate::Color::yellow).font_style({tabulate::FontStyle::bold});

    table.add_row({"Funciton Name", "Duration / ms"});
    table[1].format().font_color(tabulate::Color::yellow).font_style({tabulate::FontStyle::bold});

    bool first_data_row = false;
    int row_idx = 2;
    for (std::size_t idx = 0; idx < names_.size(); ++idx)
    {
        std::string duration_str = std::to_string(durations_[idx].count());
        if (durations_[idx] == 999999ms)
            duration_str = "Execution Error";

        table.add_row({names_[idx], std::to_string(durations_[idx].count())});
        table[row_idx].format().font_color(tabulate::Color::green);
        if (duration_str == "Execution Error")
            table[row_idx][1].format().font_background_color(tabulate::Color::red);

        if (first_data_row)
            table[row_idx].format().hide_border_top();

        first_data_row = true;
        ++row_idx;
    }

    stream << table << std::endl;
}

} // namespace timer