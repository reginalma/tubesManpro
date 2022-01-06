<?php
    // session_start();
    // print_r($result);
    // echo $_SESSION['kota'];
    // echo $_GET['tanggal'];
?>
<html>
    <head>
        <title>GeoData</title>
        <link href="https://netdna.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
		<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
		<script src="https://netdna.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js"></script>
		<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.4.1/css/all.css" integrity="sha384-5sAR7xN1Nv6T6+dT2mhtzEpVJvfS3NScPQTrOxhwjIuvcA67KV2R5Jz6kr4abQsz" crossorigin="anonymous">
		<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
		<link rel="stylesheet" href="https://www.w3schools.com/lib/w3-colors-flat.css">
        <link rel='stylesheet' href='https://fonts.googleapis.com/css?family=Roboto'>
		<link rel="stylesheet" href="view/style/style.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <?php
            if(isset($start_session)) {
                $start_session = session_start();
            }

            $array_kota = array(
                'Albury', 
                'BadgerysCreek', 
                'Cobar', 
                'CoffsHarbour', 
                'Moree',
                'Newcastle', 
                'NorahHead', 
                'NorfolkIsland', 
                'Penrith', 
                'Richmond',
                'Sydney', 
                'SydneyAirport', 
                'WaggaWagga', 
                'Williamtown',
                'Wollongong', 
                'Canberra', 
                'Tuggeranong', 
                'MountGinini', 
                'Ballarat',
                'Bendigo', 
                'Sale', 
                'MelbourneAirport', 
                'Melbourne', 
                'Mildura',
                'Nhil', 
                'Portland', 
                'Watsonia', 
                'Dartmoor', 
                'Brisbane', 
                'Cairns',
                'GoldCoast', 
                'Townsville', 
                'Adelaide', 
                'MountGambier', 
                'Nuriootpa',
                'Woomera', 
                'Albany', 
                'Witchcliffe', 
                'PearceRAAF', 
                'PerthAirport',
                'Perth', 
                'SalmonGums', 
                'Walpole', 
                'Hobart', 
                'Launceston',
                'AliceSprings', 
                'Darwin', 
                'Katherine', 
                'Uluru'
            );
        ?>
        <div class="w3-container header">
            <h1 style="margin: 0px; margin-bottom: -37px; padding-top: 20px; padding-left: 20px">
                <a href="mainPage" class="mainPageLink">
                    GeoData
                </a>
            </h1>
            <div class="w3-bar" style="padding-right: 20px">
                <button class="w3-bar-item w3-button w3-right buttonPrakira">
                    <h4><a href="prakiraCuaca" id="menuPrakira">Prakira Cuaca</a></h4>
                </button>
                <button class="w3-bar-item w3-button w3-right buttonHistory">
                    <div class="w3-dropdown-click hist">
                        <h4 onclick="dropHistory()" >History Cuaca</h4>
                        <div id="Demo" class="w3-dropdown-content w3-bar-block w3-border buttonHistoryContent">
                            <?php
                                // $_SESSION['kota'] = $_GET['kota']
                                foreach($array_kota as $k) {
                                    echo "<a href='historyCuaca?kota=$k' class='w3-bar-item w3-button'>$k</a>";
                                }
                            ?>
                        </div>
                    </div>
                </button>
            </div>
		</div>
        <br>
        <div class="cuaca">
            <h1>
                <?php
                    if(isset($_GET['kota'])) {
                        $kota_history = $_GET['kota'];
                        $_SESSION['kota'] = $_GET['kota'];
                        echo $kota_history;
                    }
                    else {
                        $kota_history = $_SESSION['kota'];
                        echo $kota_history;
                    }         
                ?>
            </h1>
        </div>
        <div>
            <form action="#" method="GET">
                <input type="date" class="tanggal" name="tanggal" value="<?php echo date('Y-m-d') ?>">
                <input type="date" class="tanggal2" name="tanggal_from" value="<?php echo date('Y-m-d') ?>">
                <input type="date" class="tanggal3" name="tanggal_to" value="<?php echo date('Y-m-d') ?>">
                <button href="#" id="btnSearch" class="w3-btn">Search</button>
            </form>
        </div>
        <br>
        <?php
            if (isset($_GET['tanggal'])) {
                $tanggalH = date('Y-m-d', strtotime($_GET['tanggal']));
            }
            if (isset($_GET['tanggal_from'])) {
                $tanggal_from = date('Y-m-d', strtotime($_GET['tanggal_from']));
            }
            if (isset($_GET['tanggal_to'])) {
                $tanggal_to = date('Y-m-d', strtotime($_GET['tanggal_to']));
            }
        ?>
        <div class="container_hist">
            <div id="formPredik" style="height: 540px">
                <form action="" method="">
                    <table style="margin-top: -20px">
                        <?php
                            if (isset($_GET['tanggal'])) {
                                $command_history_temp_9am = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryTemp9am.py $kota_history $tanggalH");
                                $history_temp_9am = shell_exec($command_history_temp_9am);

                                $command_history_windSpeed_9am = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryWindSpeed9am.py $kota_history $tanggalH");
                                $history_windSpeed_9am = shell_exec($command_history_windSpeed_9am);
                                
                                $command_history_humidity_9am = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryHumidity9am.py $kota_history $tanggalH");
                                $history_humidity_9am = shell_exec($command_history_humidity_9am);

                                $command_history_rainfall_9am = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryRainfall9am.py $kota_history $tanggalH");
                                $history_rainfall_9am = shell_exec($command_history_rainfall_9am);

                                $command_history_pressure_9am = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryPressure9am.py $kota_history $tanggalH");
                                $history_pressure_9am = shell_exec($command_history_pressure_9am);

                                $command_history_windDir_9am = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryWindDir9am.py $kota_history $tanggalH");
                                $history_windDir_9am = shell_exec($command_history_windDir_9am);
                            }
                        ?>
                        <tr>
                            <td>
                                Suhu :
                                <?php 
                                    if (isset($history_temp_9am)) {
                                        echo $history_temp_9am; 
                                    }
                                ?>
                                &deg;
                            </td>
                        </tr>   
                        <tr>
                            <td>
                                Kecepatan Angin :
                                <?php 
                                    if (isset($history_windSpeed_9am)) {
                                        echo $history_windSpeed_9am; 
                                    }
                                ?>
                                km/h
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Kelembapan :
                                <?php 
                                    if (isset($history_humidity_9am)) {
                                        echo $history_humidity_9am; 
                                    }
                                ?>
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Curah Hujan :
                                <?php 
                                    if (isset($history_rainfall_9am)) {
                                        echo $history_rainfall_9am; 
                                    }
                                ?>
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Tekanan :
                                <?php 
                                    if (isset($history_pressure_9am)) {
                                        echo $history_pressure_9am; 
                                    }
                                ?>
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Arah Angin : 
                                <?php 
                                    if (isset($history_windDir_9am)) {
                                        echo $history_windDir_9am; 
                                    }
                                ?>
                            </td>
                        </tr>
                    </table>
                </form>
            </div>
            <div id="formPredik" style="height: 540px">
                <form action="" method="">
                    <table style="margin-top: -20px">
                    <?php
                            if (isset($_GET['tanggal'])) {
                                $command_history_temp_3pm = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryTemp3pm.py $kota_history $tanggalH");
                                $history_temp_3pm = shell_exec($command_history_temp_3pm);

                                $command_history_windSpeed_3pm = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryWindSpeed3pm.py $kota_history $tanggalH");
                                $history_windSpeed_3pm = shell_exec($command_history_windSpeed_3pm);
                                
                                $command_history_humidity_3pm = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryHumidity3pm.py $kota_history $tanggalH");
                                $history_humidity_3pm = shell_exec($command_history_humidity_3pm);

                                $command_history_rainfall_3pm = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryRainfall3pm.py $kota_history $tanggalH");
                                $history_rainfall_3pm = shell_exec($command_history_rainfall_3pm);

                                $command_history_pressure_3pm = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryPressure3pm.py $kota_history $tanggalH");
                                $history_pressure_3pm = shell_exec($command_history_pressure_3pm);

                                $command_history_windDir_3pm = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryWindDir3pm.py $kota_history $tanggalH");
                                $history_windDir_3pm = shell_exec($command_history_windDir_3pm);
                            }
                        ?>
                        <tr>
                            <td>
                                Suhu :
                                <?php 
                                    if (isset($history_temp_3pm)) {
                                        echo $history_temp_3pm; 
                                    }
                                ?>
                                &deg;
                            </td>
                        </tr>   
                        <tr>
                            <td>
                                Kecepatan Angin :
                                <?php 
                                    if (isset($history_windSpeed_3pm)) {
                                        echo $history_windSpeed_3pm; 
                                    }
                                ?>
                                km/h
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Kelembapan :
                                <?php 
                                    if (isset($history_humidity_3pm)) {
                                        echo $history_humidity_3pm; 
                                    }
                                ?>
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Curah Hujan :
                                <?php 
                                    if (isset($history_rainfall_3pm)) {
                                        echo $history_rainfall_3pm; 
                                    }
                                ?>
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Tekanan :
                                <?php 
                                    if (isset($history_pressure_3pm)) {
                                        echo $history_pressure_3pm; 
                                    }
                                ?>
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Arah Angin : 
                                <?php 
                                    if (isset($history_windDir_3pm)) {
                                        echo $history_windDir_3pm; 
                                    }
                                ?>
                            </td>
                        </tr>
                    </table>
                </form>
            </div>
            <div id="grafik1" style="height: 540px">
                <h2 style="text-align: center; margin: 25px 0px 15px 0px;">Grafik</h2>
                <canvas id="myChart"></canvas>
                <?php
                    if (isset($_GET['tanggal_from'])) {
                        if(isset($_GET['tanggal_to'])) {
                            $command_history_temp = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryTemp9am.py $kota_history $tanggal_from $tanggal_to");
                            $history_temp = shell_exec($command_history_temp);
                            
                            $command_history_humidity= escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryGraphHumidity.py $kota_history $tanggal_from $tanggal_to");
                            $history_humidity = shell_exec($command_history_humidity);

                            $command_history_rainfall = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryRainfall9am.py $kota_history $tanggal_from $tanggal_to");
                            $history_rainfall = shell_exec($command_history_rainfall);
                        }
                    }
                ?>
                <script>
                    var xValues = [100,200,300,400,500,600,700,800,900,1000];
                    new Chart("myChart", {
                        type: "line",
                        data: {
                            labels: xValues,
                            datasets: [{ 
                            label: "Rainfall",
                            data: [860,1140,1060,1060,1070,1110,1330,2210,7830,2478],
                            borderColor: "red",
                            fill: false
                            }, { 
                            label: "Temp",
                            data: [1600,1700,1700,1900,2000,2700,4000,5000,6000,7000],
                            borderColor: "green",
                            fill: false
                            }, { 
                            label: "Humidity",
                            data: [300,700,2000,5000,6000,4000,2000,1000,200,100],
                            borderColor: "blue",
                            fill: false
                            }]
                        },
                        options: {
                            legend: {display: true}
                        }
                    }); 
                </script>
            </div>
            <a href="GeoData_MainPage.html" id="btnHome" class="w3-btn" style="margin-top: 100px">Home</a>
        </div>
        <script>
            function dropHistory() {
                var x = document.getElementById("Demo");
                if (x.className.indexOf("w3-show") == -1) {
                    x.className += " w3-show";
                } else { 
                    x.className = x.className.replace(" w3-show", "");
                }
            }
        </script>
    </body>
</html>