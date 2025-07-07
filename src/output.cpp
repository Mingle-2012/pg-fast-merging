#include "output.h"

CsvLogger::CsvLogger(const std::string& filePath, int precision)
    : filePath_(filePath), headerWritten_(false), precision_(precision) {
    ofs_.open(filePath_, std::ios::out | std::ios::app);
    if (!ofs_.is_open()) {
        std::cerr << "Error: CsvLogger - Could not open file " << filePath_ << std::endl;
    }
}

CsvLogger::~CsvLogger() {
    if (ofs_.is_open()) {
        ofs_.close();
    }
}

bool
CsvLogger::writeHeader(const std::vector<std::string>& headers) {
    if (!ofs_.is_open()) {
        std::cerr << "Error: CsvLogger - Cannot write header, file not open." << std::endl;
        return false;
    }
    if (headerWritten_) {
        return true;
    }

    for (size_t i = 0; i < headers.size(); ++i) {
        ofs_ << toCsvString(headers[i]);
        if (i < headers.size() - 1) {
            ofs_ << ",";
        }
    }
    ofs_ << "\n";
    ofs_.flush();
    headerWritten_ = true;
    return true;
}

bool
CsvLogger::isOpen() const {
    return ofs_.is_open();
}

std::string
CsvLogger::toCsvString(const std::string& value) {
    std::string escaped_value = value;
    if (escaped_value.find(',') != std::string::npos ||
        escaped_value.find('\n') != std::string::npos ||
        escaped_value.find('"') != std::string::npos) {
        size_t pos = escaped_value.find('"');
        while (pos != std::string::npos) {
            escaped_value.replace(pos, 1, "\"\"");
            pos = escaped_value.find('"', pos + 2);
        }
        escaped_value = "\"" + escaped_value + "\"";
    }
    return escaped_value;
}
