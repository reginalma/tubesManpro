<?php 
    $url = $_SERVER['REDIRECT_URL'];
    $baseURL = '/tubesManpro';

    if($_SERVER["REQUEST_METHOD"] == "GET"){
        switch($url){
            case $baseURL.'/index':
                require_once "control/indexController.php";
                // $idxCtrl = new indexController();
                echo $idxCtrl->view_mainpage();
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