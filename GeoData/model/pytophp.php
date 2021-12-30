<?php
    $command = escapeshellcmd('python3 /Tubes_ManPro.py');
    $output = shell_exec($command);
    echo $output;
?>