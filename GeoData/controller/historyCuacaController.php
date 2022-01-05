<?php
    require_once 'controller/service/view.php';
    include('service/mysqlDB.php');
    include 'model/weatheraus.php';
    session_start();
    class historyCuacaController {
        function __construct()
        {
            $this->db = new MySQLDB("localhost", "root", "", "weatheraus");
        }

        public function start() {
            if (isset($_GET['tanggal'])){
                $kota = $_SESSION['kota'];
                $tanggal = $_GET['tanggal'];
                $result= $this-> exe($kota,$tanggal);
                return View::createView('historyCuaca.php', ["result" => $result]);
            }
            else{
                return View::createView('historyCuaca.php',[]);
            }
            // $kota = $_SESSION['kota'];
            // $tanggal = $_GET['tanggal'];
            // $result= $this-> exe($kota,$tanggal);
            // return View::createView('historyCuaca.php', ["result" => $result]
    
        }
        
        public function exe($kota, $tanggal){
            $query= "SELECT * FROM weatheraus_clean WHERE Location='$kota' AND Date ='$tanggal'";
            $select = $this->db->executeSelectQuery($query);
            $result = [];
            foreach ($select as $value) {
                $result[] = new weatheraus(
                    $value['Date'],
                    $value['Location'],
                    $value['MinTemp'],
                    $value['MaxTemp'],
                    $value['Rainfall'],
                    $value['Evaporation'],
                    $value['Sunshine'],
                    $value['WindGustDir'],
                    $value['WindGustSpeed'],
                    $value['WindDir9am'],
                    $value['WindDir3pm'],
                    $value['WindSpeed9am'],
                    $value['WindSpeed3pm'],
                    $value['Humidity9am'],
                    $value['Humidity3pm'],
                    $value['Pressure9am'],
                    $value['Pressure3pm'],
                    $value['Cloud9am'],
                    $value['Cloud3pm'],
                    $value['Temp9am'],
                    $value['Temp3pm'],
                    $value['RainToday'],
                    $value['RainTomorrow']
                );
            }
            return $result;
        }
    }
?>