<?php
    $command = escapeshellcmd('python ..\\GeoData\\model\\testPrintData.py');
    $output = shell_exec($command);
    echo $output;
    // // $output = passthru("python testPrintData.py")


    // // $res = shell_exec('python D:\UtilityApp\xampp\htdocs\GeoData\model\testPrintData.py', 'param1', 'param2');
    // // echo $res;
?>