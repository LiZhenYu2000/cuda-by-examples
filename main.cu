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
    int ch_5_1(void);
    int ch_5_2(void);
}
//collection of chapter6 codes
namespace ch6 {
    int ch_6(void);
}

int main(int argc, char* argv[]) {
    if(argc <= 1) {
        std::cerr << "Usage: .\\Main.exe <lab_number>";
        return -1;
    }

    // assert(argc >= 2)
    std::istringstream strNum(std::string{argv[1]});
    int labNum = -1;
    int subLabNum = -1;
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
        if(strNum >> subLabNum) {
            if(subLabNum == 1)
                stat = ch5::ch_5_1();
            else if(subLabNum == 2)
                stat = ch5::ch_5_2();
            else
                return -1;
        } else
            return -1;
        break;
    case 6:
        stat = ch6::ch_6();
        break;
    default:
        std::cerr << "Usage: <lab_number> not exist";
        return -1;
    }

    return stat;
}