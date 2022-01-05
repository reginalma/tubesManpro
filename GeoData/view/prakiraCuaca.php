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
    </head>
    <body>
        <?php
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
                Prakira Cuaca
            </h1>
        </div>
        <br>
        <div class="container">
            <div id="formPredik" style="height: 540px; padding-bottom: 20px">
                <form action="" method="GET">
                    <table>
                        <tr>
                            <td>
                                Rainfall
                            </td>
                            <td>
                                <input type="text" name="rainfall">
                            </td>
                        </tr>   
                        <tr>
                            <td>
                                Sunshine
                            </td>
                            <td>
                                <input type="text" name="sunshine">
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Humidity 9am
                            </td>
                            <td>
                                <input type="text" name="humidity9am">
                            </td>
                        </tr> 
                        <tr>
                            <td>
                                Humidity 3pm
                            </td>
                            <td>
                                <input type="text" name="humidity3pm">
                            </td>
                        </tr>
                        <tr>
                            <td>
                                Waktu Prediksi
                            </td>
                            <td>
                            <select name="hari" id="dropdownHari">
                                <option value="hari ini">Hari ini</option>
                                <option value="besok">Besok</option>
                            </select>
                            </td>
                        </tr> 
                    </table>
                    <input id="sumbitForm" type="submit" value="Submit" style="margin-top: 5px">
                </form>
            </div>
            <div id="result">
                <h1>Hasil Prediksi</h1>
                <script>
                    function getTime(){
                        console.log(document.getElementById('dropdownHari').value);
                        let time = document.getElementById('dropdownHari').value;
                        return time
                    }
                </script>
                <?php
                    if(isset($_GET["humidity9am"]) && isset($_GET["humidity3pm"]) && isset($_GET["rainfall"]) && isset($_GET["sunshine"])){
                        $hari = $_GET["hari"];
                        $a=$_GET["rainfall"];
                        $b=$_GET["sunshine"];
                        $c=$_GET["humidity9am"];
                        $d=$_GET["humidity3pm"];
                        if($hari == "hari ini"){
                            $tmp = exec("python ..\\GeoData\\predict\\predict_today.py 2>&1".$a." ".$b." ".$c." ".$d);
                            if($tmp == "Yes"){
                                echo "<img  class=\"result\" src=\"view/image/cloud-rain-icons-11033.png\" width=\"250\" height=\"230\" style=\"margin-left: 320px\">";
                                echo "<h1>Hujan</h1>";  
                            }else if($tmp == "No"){
                                echo "<img class=\"result\" src=\"view/image/sun.png\" width=\"250\" height=\"250\" style=\"margin-left: 320px\">";
                                echo "<h1>Cerah</h1>";  
                            }
                        }
                        else{
                            $tmp = exec("python ..\\GeoData\\predict\\predict_tomorrow.py 2>&1".$a." ".$b." ".$c." ".$d);
                            // if($tmp == "1"){
                            //     echo "<img  class=\"result\" src=\"view/image/thunderstorms.png\" width=\"250\" height=\"230\">";
                            //     echo "<h1>Hujan</h1>";  
                            // }else if($tmp == "0"){
                            //     echo "<img class=\"result\" src=\"view/image/sun.png\" width=\"250\" height=\"250\">";
                            //     echo "<h1>Cerah</h1>";  
                            // }
                            if($tmp == "Yes"){
                                echo "<img  class=\"result\" src=\"view/image/cloud-rain-icons-11033.png\" width=\"250\" height=\"230\" style=\"margin-left: 320px\">";
                                echo "<h1>Hujan</h1>";  
                            }else if($tmp == "No"){
                                echo "<img class=\"result\" src=\"view/image/sun.png\" width=\"250\" height=\"250\" style=\"margin-left: 320px\">";
                                echo "<h1>Cerah</h1>";  
                            }
                        }
                    }
                ?>
            </div>
            <a href="mainPage" id="btnHome" class="w3-btn" style="margin-top: 100px">
                Home
            </a>
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