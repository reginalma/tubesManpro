<?php
    $command = escapeshellcmd('C:\Users\RelStrixG\AppData\Local\Programs\Python\Python310\python.exe D:\UtilityApp\xampp\htdocs\GeoData\model\testPrintData.py');
    $output = shell_exec($command);
    echo $output;
    // $output = passthru("python testPrintData.py")
    
    echo "<img src='image/partly_coudly.png' alt='partly' style='width: 100%'>";
?>