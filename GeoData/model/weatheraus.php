<?php 
    class weatheraus{
        private $date;
        private $location;
        private $minTemp;
        private $maxTemp;
        private $rainfall; 
        private $evaporation;
        private $sunshine;
        private $windGustDir;
        private $windGustSpeed;
        private $windDir9am;
        private $windDir3pm;
        private $windSpeed9am;
        private $windSpeed3pm;
        private $humidity9am;
        private $humidity3pm;
        private $pressure9am;
        private $pressure3pm;
        private $cloud9am;
        private $cloud3pm;
        private $temp9am;
        private $temp3pm;
        private $rainToday;
        private $rainTomorrow;


        function __construct($date,$location,$minTemp,$maxTemp,$rainfall,$evaporation,$sunshine,$windGustDir,$windGustSpeed,$windDir9am,
        $windDir3pm,$windSpeed9am,$windSpeed3pm,$humidity9am,$humidity3pm,$pressure9am,$pressure3pm,$cloud9am,$cloud3pm,$temp9am,$temp3pm
        ,$rainToday,$rainTomorrow){
            $this->date = $date;
            $this->location = $location;
            $this->minTemp = $minTemp;
            $this->maxTemp = $maxTemp;
            $this->rainfall = $rainfall;
            $this->evaporation = $evaporation;
            $this->sunshine = $sunshine;
            $this->windGustDir = $windGustDir;
            $this->windGustSpeed = $windGustSpeed;
            $this->windDir9am = $windDir9am;
            $this->windDir3pm=$windDir3pm;
            $this->windSpeed3pm=$windSpeed3pm;
            $this->windSpeed9am=$windSpeed9am;
            $this->humidity9am=$humidity9am;
            $this->humidity3pm=$humidity3pm;
            $this->pressure9am=$pressure9am;
            $this->pressure3pm=$pressure3pm;
            $this->cloud9am=$cloud9am;
            $this->cloud3pm=$cloud3pm;
            $this->temp9am=$temp9am;
            $this->temp3pm=$temp3pm;
            $this->rainToday=$rainToday;
            $this->rainTomorrow=$rainTomorrow;
        }

    public function getDate (){
        return $this->date;
    }

    
    public function getLocation (){
        return $this->location;
    }

    
    public function getMinTemp (){
        return $this->minTemp;
    }
    
    public function getMaxTemp (){
        return $this->maxTemp;
    }

    public function getRainfall(){
        return $this->rainfall;
    }
    public function getEvaporation(){
        return $this->evaporation;
    }
    public function getSunshine(){
        return $this->sunshine;
    }

    public function getWindgustdir(){
        return $this -> windGustDir;
    }

    public function getWindgustspeed(){
        return $this-> windGustspeed;
    }

    public function getWinddir9am(){
        return $this-> windDir9am;
    }

    public function getWinddir3pm(){
        return $this-> windSpeed3pm;
    }

    public function getWindspeed9am(){
        return $this-> windSpeed9am;
    }
    
    public function getWindspeed3pm(){
        return $this-> windSpeed3pm;
    }

    public function getHumidity9am(){
        return $this-> humidity9am;
    }

    public function getHumidity3pm(){
        return $this-> humidity3pm;
    }

    public function getPressure9am(){
        return $this-> pressure9am;
    }

    public function getPressure3pm(){
        return $this-> pressure3pm;
    }

    public function getCloud9am(){
        return $this-> cloud9am;
    }

    public function getCloud3pm(){
        return $this-> cloud3pm;
    }


    public function getTemp9am(){
        return $this-> temp9am;
    }

    public function getTemp3pm(){
        return $this-> temp3pm;
    }

    public function getRaintoday(){
        return $this-> rainToday;
    }

    public function getRaintomorrow(){
        return $this-> rainTomorrow;
    }
    }

?>