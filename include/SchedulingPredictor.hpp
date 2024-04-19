#ifndef SCHEDULING_PREDICTOR_HPP
#define SCHEDULING_PREDICTOR_HPP

#include <vector>
#include <map>
#include <queue>

// Class to predict thread block scheduling on SMs
class SchedulingPredictor {
    SchedulingPredictor(int num_sms);
    ~SchedulingPredictor();
}

#endif // SCHEDULING_PREDICTOR_HPP