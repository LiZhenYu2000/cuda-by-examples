#include <iostream>
#include <sstream>
#include <book_cpp.hpp>

// collection of chapter3 codes
namespace ch3 {
    int ch_3(void);
}
// collection of chapter4 codes
namespace ch4 {
    int ch_4(void);
}
// collection of chapter5 codes
namespace ch5 {
    int ch_5(void);
}

int main(int argc, char* argv[]) {
    if(argc <= 1) {
        std::cerr << "Usage: .\\Main.exe <lab_number>";
        return -1;
    }

    // assert(argc >= 2)
    std::istringstream strNum(std::string{argv[1]});
    int labNum = -1;
    strNum >> labNum;

    int stat = 0;

    switch (labNum)
    {
    case 3:
        stat = ch3::ch_3();
        break;
    case 4:
        stat = ch4::ch_4();
        break;
    case 5:
        stat = ch5::ch_5();
        break;
    default:
        std::cerr << "Usage: <lab_number> not exist";
        return -1;
    }

    return stat;
}