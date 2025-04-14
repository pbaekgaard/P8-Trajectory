#include <trajectory.hpp>
Trajectory t(0, std::vector<SamplePoint>{
                    SamplePoint(3, 15.5, 0),     SamplePoint(5, 15.5, 5),    SamplePoint(7, 15.5, 10),
                    SamplePoint(8.5, 15.5, 15),  SamplePoint(9.5, 15.5, 20), SamplePoint(10, 15.5, 25),
                    SamplePoint(11.5, 15.5, 30), SamplePoint(12, 14, 35),    SamplePoint(12, 12, 40),
                    SamplePoint(12, 11, 45),     SamplePoint(12, 10, 50),    SamplePoint(12, 8, 55),
                    SamplePoint(12, 5.5, 60),    SamplePoint(13, 4, 65),     SamplePoint(14, 3, 70),
                    SamplePoint(14, 2, 75),      SamplePoint(16, 2, 80),     SamplePoint(18.5, 2, 85),
                    SamplePoint(20.5, 2, 90),    SamplePoint(21.5, 2, 95)});

Trajectory t1(1,
              std::vector<SamplePoint>{SamplePoint(2, 2.5, 0), SamplePoint(1.5, 3, 5), SamplePoint(1.5, 4, 10),
                                       SamplePoint(1.5, 5.5, 15), SamplePoint(1.5, 7, 20), SamplePoint(1.5, 8.5, 25),
                                       SamplePoint(1.5, 9.5, 30), SamplePoint(1.5, 10.5, 35), SamplePoint(1.5, 12, 40),
                                       SamplePoint(1.5, 13, 45), SamplePoint(2, 14, 50), SamplePoint(3, 14.55, 55),
                                       SamplePoint(5, 14.55, 60), SamplePoint(7, 14.55, 65)});

Trajectory t2(2, std::vector<SamplePoint>{
                     SamplePoint(5, 16, 0), SamplePoint(7.5, 16, 5), SamplePoint(8.5, 16, 10), SamplePoint(9.5, 16, 15),
                     SamplePoint(12, 15.5, 20), SamplePoint(12.5, 14.5, 25), SamplePoint(12.5, 13.5, 30),
                     SamplePoint(12.5, 12, 35), SamplePoint(14, 11.5, 40), SamplePoint(15, 11, 45),
                     SamplePoint(16.5, 11, 50), SamplePoint(17.5, 11, 55), SamplePoint(18.5, 11, 60),
                     SamplePoint(19.5, 11.5, 65), SamplePoint(19.5, 12, 70), SamplePoint(19.5, 13.5, 75),
                     SamplePoint(19.5, 15, 80)});

Trajectory t3(3, std::vector<SamplePoint>{SamplePoint(5.5, 14, 0), SamplePoint(7, 14, 5), SamplePoint(8, 14.5, 10),
                                          SamplePoint(10, 14.5, 15), SamplePoint(11, 14, 20), SamplePoint(11.5, 12, 25),
                                          SamplePoint(9.5, 12, 30), SamplePoint(8, 12, 35), SamplePoint(6, 12, 40),
                                          SamplePoint(4.5, 12, 45), SamplePoint(3.5, 12, 50), SamplePoint(2.5, 11, 55),
                                          SamplePoint(2, 10.5, 60), SamplePoint(2, 9, 65), SamplePoint(2, 8, 70)});

Trajectory t4(4,
              std::vector<SamplePoint>{SamplePoint(5.58, 11, 0), SamplePoint(6.58, 11, 5), SamplePoint(7.57, 11, 10),
                                       SamplePoint(8.58, 11, 15), SamplePoint(9.55, 11, 20), SamplePoint(11.5, 11, 25),
                                       SamplePoint(11.5, 10, 30), SamplePoint(11.5, 8.5, 35), SamplePoint(11.5, 8, 40),
                                       SamplePoint(11.5, 6, 45), SamplePoint(11.5, 5.11, 50),
                                       SamplePoint(12.5, 4.1, 55), SamplePoint(13, 3.5, 60), SamplePoint(13.5, 3, 65),
                                       SamplePoint(13.5, 2, 70), SamplePoint(13.5, 1, 75), SamplePoint(12.5, 1, 80),
                                       SamplePoint(10.5, 1, 85), SamplePoint(8.5, 1, 90)});

Trajectory t5(5, std::vector<SamplePoint>{SamplePoint(20.5, 13, 0), SamplePoint(19, 13, 5), SamplePoint(17.5, 13, 10),
                                          SamplePoint(16, 13, 15), SamplePoint(15, 13, 20), SamplePoint(14, 12.5, 25),
                                          SamplePoint(12.5, 11, 30), SamplePoint(12.5, 10, 35),
                                          SamplePoint(12.5, 8, 40), SamplePoint(12.5, 6, 45),
                                          SamplePoint(13.5, 4.5, 50), SamplePoint(14.5, 3.5, 55)});

Trajectory t6(6,
              std::vector<SamplePoint>{SamplePoint(2, 6, 0), SamplePoint(2, 4.5, 5), SamplePoint(2, 3.5, 10),
                                       SamplePoint(2.5, 2.5, 15), SamplePoint(3.5, 2.5, 20), SamplePoint(4.5, 2.5, 25),
                                       SamplePoint(5.5, 2.5, 30), SamplePoint(6.5, 2.5, 35), SamplePoint(7.5, 2.5, 40),
                                       SamplePoint(9, 2.5, 45), SamplePoint(11, 2.5, 50), SamplePoint(12, 2.5, 55),
                                       SamplePoint(14.5, 2.5, 60), SamplePoint(16, 2.5, 65)});

Trajectory t7(7, std::vector<SamplePoint>{SamplePoint(15.5, 6.5, 0), SamplePoint(15.5, 6, 5), SamplePoint(15.5, 5, 10),
                                          SamplePoint(16.5, 5, 15), SamplePoint(17.5, 5, 20), SamplePoint(17.5, 4, 25),
                                          SamplePoint(18, 3, 30), SamplePoint(18, 2.5, 35), SamplePoint(20, 2.5, 40),
                                          SamplePoint(22, 2.5, 45)});
