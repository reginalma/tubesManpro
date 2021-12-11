<?php 
    $url = $_SERVER['REDIRECT_URL'];
    $baseURL = '/GeoData';

    if($_SERVER["REQUEST_METHOD"] == "GET"){
        switch($url){
            case $baseURL."/mainPage":
                require_once "controller/mainPageController.php";
                $main = new mainPageController();
                echo $main->start();
                break;
            case $baseURL."/prakiraCuaca":
                require_once "controller/prakiraCuacaController.php";
                $prakira = new prakiraCuacaController();
                echo $prakira->start();
                break;
        
            default :
                echo '404 not found';
                break;
        }
    }else if($_SERVER["REQUEST_METHOD"] == "POST"){
        switch($url){
                
            default :
                echo '404 not found';
                break;
        }
    }
?>