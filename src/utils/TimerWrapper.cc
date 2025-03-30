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
  using namespace tabulate;
  using Row_t = Table::Row_t;

  Table wrapper_name;
  wrapper_name.format().locale("zh_CN.UTF-8").hide_border().width(60);
  wrapper_name.add_row(Row_t{wrapper_name_});
  wrapper_name[0].format().font_align(FontAlign::center).font_color(Color::yellow).font_style({FontStyle::bold, FontStyle::italic});
  std::cout << wrapper_name;

  Table content;
  content.format().locale("zh_CN.UTF-8").border_color(Color::magenta);

  content.add_row({"Funciton Name", "Duration / ms"});
  content[0].format().font_color(Color::yellow).font_style({FontStyle::bold, FontStyle::italic});

  bool first_data_row = false;
  int row_idx = 1;
  for (std::size_t idx = 0; idx < names_.size(); ++idx)
  {
    std::string duration_str = std::to_string(durations_[idx].count());
    if (durations_[idx] == 999999ms)
      duration_str = "Execution Error";

    content.add_row({names_[idx], std::to_string(durations_[idx].count())});
    content[row_idx].format().font_color(Color::green).font_style({FontStyle::italic});
    if (duration_str == "Execution Error")
      content[row_idx][1].format().font_background_color(Color::red);

    if (first_data_row)
      content[row_idx].format().hide_border_top();

    first_data_row = true;
    ++row_idx;
  }

  content.format().hide_border_top().border_color(Color::magenta);
  content.column(0).format().font_align(FontAlign::center).width(40);
  content.column(1).format().font_align(FontAlign::center).width(20);

  std::cout << content << std::endl;
}

} // namespace timer