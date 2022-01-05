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
            session_start();

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
            <form action="historyCuaca?kota=$kota_history" method="GET">
                <input type="date" class="tanggal" name="tanggal" value="<?php echo date('Y-m-d') ?>">
            
            
                <input type="date" class="tanggal2">
                <input type="date" class="tanggal3">
                <button href="#" id="btnSearch" class="w3-btn">Search</button>
            </form>
        </div>
        <br>
        <?php
            if (isset($_GET['tanggal'])) {
                $tanggalH = date('Y-m-d', strtotime($_GET['tanggal']));
            }
        ?>
        <div class="container_hist">
            <div id="formPredik" style="height: 540px">
                <form action="" method="">
                    <table style="margin-top: -20px">
                        <?php
                            if (isset($_GET['tanggal'])) {
                                $command_history_temp = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryTemp.py, $kota_history $tanggalH");
                                // echo $command_history_temp;
                                // $command_history_temp = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryTemp.py $kota_history");
                                // $command_history_temp = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryTemp.py $tanggalH");
                                $history_temp = shell_exec($command_history_temp);

                                $command_history_windSpeed = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryWindSpeed.py $kota_history");
                                $history_windSpeed = shell_exec($command_history_windSpeed);
                                
                                $command_history_humidity = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryHumidity.py $kota_history");
                                $history_humidity = shell_exec($command_history_humidity);

                                $command_history_rainfall = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryRainfall.py $kota_history");
                                $history_rainfall = shell_exec($command_history_rainfall);

                                $command_history_pressure = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryPressure.py $kota_history");
                                $history_pressure = shell_exec($command_history_pressure);

                                $command_history_windDir = escapeshellcmd("python ..\\GeoData\\model\\WeatherHistoryWindDir.py $kota_history");
                                $history_windDir = shell_exec($command_history_windDir);
                            }
                        ?>
                        <tr>
                            <td>
                                Suhu :
                                <?php 
                                    if (isset($history_temp)) {
                                        echo $history_temp; 
                                    }
                                ?>
                                &deg;C
                            </td>
                        </tr>   
                        <tr>
                            <td>
                                Kecepatan Angin :
                                <?php 
                                    if (isset($history_windSpeed)) {
                                        echo $history_windSpeed; 
                                    }
                                ?>
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Kelembapan :
                                <?php 
                                    if (isset($history_humidity)) {
                                        echo $history_humidity; 
                                    }
                                ?>
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Curah Hujan :
                                <?php 
                                    if (isset($history_rainfall)) {
                                        echo $history_rainfall; 
                                    }
                                ?>
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Tekanan :
                                <?php 
                                    if (isset($history_pressure)) {
                                        echo $history_pressure; 
                                    }
                                ?>
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Arah Angin : 
                                <?php 
                                    if (isset($history_windDir)) {
                                        echo $history_windDir; 
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
                        <tr>
                            <td>
                                Suhu
                            </td>
                        </tr>   
                        <tr>
                            <td>
                                Kecepatan Angin
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Kelembapan
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Curah Hujan
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Tekanan
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Arah Angin
                            </td>
                        </tr> 
                    </table>
                </form>
            </div>
            <div id="grafik1" style="height: 540px">
                <h2 style="text-align: center; margin: 25px 0px 15px 0px;">Grafik</h2>
                <canvas id="myChart"></canvas>
                <script>
                    var xValues = [100,200,300,400,500,600,700,800,900,1000];

                    new Chart("myChart", {
                    type: "line",
                    data: {
                        labels: xValues,
                        datasets: [{ 
                        label: "curah hujan",
                        // data: [860,1140,1060,1060,1070,1110,1330,2210,7830,2478],
                        data: [50,50,50,50,1070,1110,1330,50,50,50],
                        borderColor: "red",
                        fill: false
                        }, { 
                        data: [1600,1700,1700,1900,2000,2700,4000,5000,6000,7000],
                        borderColor: "green",
                        fill: false
                        }, { 
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

                <form action="" method="">
                    <!-- <table>
                        
                    </table> -->
                </form>
            </div>

            <!-- <div id="grafik2">
                <form action="" method="">
                    <table>
                        
                    </table>
                </form>
            </div> -->
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